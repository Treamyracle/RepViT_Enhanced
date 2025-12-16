import sys
import os
import torch
import torch.nn as nn
import coremltools as ct
from argparse import ArgumentParser
from timm import create_model
import copy

# Setup path
sys.path.append(os.getcwd())
try:
    import repvit_sam.modeling
    from repvit_sam.modeling.repvit import RepViTBlock, RepVGGDW, Conv2d_BN, Residual
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    import repvit_sam.modeling
    from repvit_sam.modeling.repvit import RepViTBlock, RepVGGDW, Conv2d_BN, Residual

def count_layers(model):
    conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    bn = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    return conv, bn

def fuse_sequential(seq_block):
    """
    Melakukan fuse pada setiap item di dalam nn.Sequential
    """
    new_layers = []
    for layer in seq_block:
        if isinstance(layer, Conv2d_BN):
            # Fuse Conv+BN menjadi Conv2d tunggal
            new_layers.append(layer.fuse())
        elif isinstance(layer, RepVGGDW):
            # Fuse RepVGGDW menjadi Conv2d tunggal
            new_layers.append(layer.fuse())
        else:
            new_layers.append(layer)
    return nn.Sequential(*new_layers)

def proper_fuse_repvit(model):
    """
    Mengganti struktur RepViTBlock dengan versi yang sudah di-fuse.
    """
    print("🔄 Memulai Penggantian Struktur (Replacing Layers)...")
    
    # RepViT menyimpan block-nya di dalam model.features (ModuleList)
    for i, block in enumerate(model.features):
        
        # 1. Fuse Token Mixer (Biasanya Sequential)
        if hasattr(block, 'token_mixer') and isinstance(block.token_mixer, nn.Sequential):
            # Ganti token_mixer lama dengan yang baru (sudah di-fuse item-itemnya)
            block.token_mixer = fuse_sequential(block.token_mixer)

        # 2. Fuse Channel Mixer
        if hasattr(block, 'channel_mixer'):
            cm = block.channel_mixer
            
            # Kasus A: Channel Mixer dibungkus Residual
            if isinstance(cm, Residual):
                # Kita tidak bisa fuse Residual-nya (karena ada GELU di dalamnya),
                # tapi kita bisa fuse Sequential di DALAM Residual.
                if isinstance(cm.m, nn.Sequential):
                    cm.m = fuse_sequential(cm.m)
            
            # Kasus B: Channel Mixer langsung Sequential
            elif isinstance(cm, nn.Sequential):
                block.channel_mixer = fuse_sequential(cm)

    return model

parser = ArgumentParser()
parser.add_argument('--resolution', default=1024, type=int)
parser.add_argument('--samckpt', default='weights/repvit_sam.pt', type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    print("\n🛠️ PROPER FUSION EXPORT 🛠️")
    
    # 1. Load Model
    model = create_model('repvit')
    
    ckpt = args.samckpt
    if not os.path.exists(ckpt): ckpt = os.path.join("..", ckpt)
    print(f"Loading: {ckpt}")
    
    state = torch.load(ckpt, map_location='cpu')
    new_state = {k.replace('image_encoder.', ''): v for k, v in state.items() if 'image_encoder' in k}
    model.load_state_dict(new_state)
    model.eval()

    c_start, bn_start = count_layers(model)
    print(f"📊 Awal: {c_start} Conv2d, {bn_start} BatchNorm2d")

    # 2. EKSEKUSI FUSE & REPLACE
    model = proper_fuse_repvit(model)

    c_end, bn_end = count_layers(model)
    print(f"📊 Akhir: {c_end} Conv2d, {bn_end} BatchNorm2d")
    
    if bn_end == 0:
        print("✅ SUKSES SEMPURNA! Semua BatchNorm hilang (Model Fully Fused).")
    elif bn_end < bn_start:
        print(f"✅ SUKSES! BatchNorm berkurang drastis ({bn_start} -> {bn_end}).")
    else:
        print("⚠️ Warning: Jumlah BatchNorm tidak berubah.")

    # 3. Export
    print("Exporting CoreML...")
    dummy_input = torch.randn(1, 3, args.resolution, args.resolution)
    traced_model = torch.jit.trace(model, dummy_input)
    
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="image", shape=dummy_input.shape)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16
    )
    
    out_path = f"coreml/repvit_{args.resolution}.mlpackage"
    mlmodel.save(out_path)
    print(f"💾 Saved: {out_path}")