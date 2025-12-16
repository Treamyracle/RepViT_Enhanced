import sys
import os
import torch
import torch.nn as nn
from timm import create_model

# Setup path
sys.path.append(os.getcwd())
try:
    import repvit_sam.modeling
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    import repvit_sam.modeling

def analyze_module(name, module):
    print(f"\n🔍 MENGANALISIS MODUL: {name}")
    print(f"   Tipe Class: {type(module)}")
    
    # Cek jumlah Conv2d di dalam modul ini (hanya direct children)
    convs = [k for k, v in module.named_children() if isinstance(v, (nn.Conv2d, nn.BatchNorm2d))]
    print(f"   Direct Conv/BN children: {len(convs)} ({convs})")
    
    # Cek atribut (variabel) yang ada
    print("   📋 Daftar Atribut Penting:")
    interesting_attrs = []
    for attr in dir(module):
        if attr.startswith('__') or attr.startswith('_'): continue
        
        val = getattr(module, attr)
        # Kita cari atribut yang berupa Module (layer) atau Method
        if isinstance(val, nn.Module):
             print(f"      - [LAYER] {attr}: {type(val).__name__}")
             interesting_attrs.append(attr)
        elif callable(val) and not isinstance(val, nn.Module):
             # Filter method bawaan PyTorch
             if attr not in dir(nn.Module()) and attr not in ['forward', 'extra_repr']:
                 print(f"      - [METHOD] {attr}")

    # Cek apakah ini terlihat seperti Block RepViT (punya banyak cabang)
    if len(interesting_attrs) > 2:
        print("   ✅ KANDIDAT BLOCK REPVIT! (Punya banyak layer)")
        return True
    return False

def main():
    print("MEMULAI DIAGNOSA STRUKTUR MODEL...")
    model = create_model('repvit')
    
    # Kita cari module pertama yang kompleks (bukan sekedar Conv2d)
    # Biasanya RepViT terdiri dari:
    # features -> Sequential -> RepViTBlock -> TokenMixer/ChannelMixer
    
    found_target = False
    
    # Telusuri features
    if hasattr(model, 'features'):
        print("Model memiliki 'features'. Menelusuri isinya...")
        for i, layer in enumerate(model.features):
            # Biasanya layer adalah RepViTBlock
            if analyze_module(f"features[{i}]", layer):
                # Jika ini block, coba lihat isinya lagi (Token Mixer?)
                for subname, sublayer in layer.named_children():
                    analyze_module(f"features[{i}].{subname}", sublayer)
                found_target = True
                break
    
    if not found_target:
        # Fallback: Scan semua module
        print("Mencari module secara brute-force...")
        for name, m in model.named_modules():
            # Cari yang punya lebih dari 1 Conv2d
            conv_count = sum(1 for c in m.children() if isinstance(c, nn.Conv2d))
            if conv_count > 1:
                analyze_module(name, m)
                break

if __name__ == "__main__":
    main()