import torch
import torch.nn as nn
import coremltools as ct
import model.repvit # Import module, bukan function langsung
import os

# --- KONFIGURASI CBAM ---
MODEL_PATH = "checkpoint_best_CBAM.pth" # Ganti dengan path weight CBAM Anda
NUM_CLASSES = 10
INPUT_SIZE = 224
OUTPUT_NAME = "RepViT_M2_3_CBAM"

# ==========================================
# 1. DEFINISI ARSITEKTUR CBAM (IN-SCRIPT)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

# ==========================================
# 2. MODIFIKASI REPVIT BLOCK (Monkey Patch)
# ==========================================
# Kita ambil class asli RepViTBlock lalu kita ganti bagian SE-nya
OriginalBlock = model.repvit.RepViTBlock

class RepViTBlockCBAM(OriginalBlock):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        # Init parent class
        super().__init__(inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs)
        
        # GANTI token_mixer jika menggunakan SE
        # Struktur token_mixer asli: [RepVGGDW/Conv, SqueezeExcite/Identity, Conv/Identity]
        # Kita cek item ke-1 (index 1) di token_mixer, biasanya itu SE
        
        if use_se:
            # Cari lokasi SE (biasanya index 1) dan ganti dengan CBAM
            # Kita asumsi struktur standard RepViT M2.3
            if isinstance(self.token_mixer, nn.Sequential):
                # Ganti layer SqueezeExcite dengan CBAM
                # inp adalah input channel layer ini
                self.token_mixer[1] = CBAM(inp) 

# --- SUNTIKKAN CLASS BARU KE MODULE ASLI ---
print("💉 Menyuntikkan CBAM ke dalam model.repvit...")
model.repvit.RepViTBlock = RepViTBlockCBAM

def convert_cbam():
    print(f"🔄 Memulai Konversi RepViT (CBAM)...")
    
    # 3. Inisialisasi Model (Sekarang otomatis pakai RepViTBlockCBAM)
    net = model.repvit.repvit_m2_3(num_classes=NUM_CLASSES)
    
    # 4. Load Weights
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: File {MODEL_PATH} tidak ditemukan.")
        return

    print(f"📂 Loading weights: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    try:
        net.load_state_dict(state_dict)
    except Exception as e:
        print(f"⚠️ Warning (Mungkin wajar jika struktur beda tipis): {e}")
        
    net.eval()
    
    # 5. Trace & Convert
    print("⚡ Tracing & Converting...")
    example_input = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
    traced_model = torch.jit.trace(net, example_input)
    
    image_input = ct.ImageType(name="image", shape=example_input.shape, scale=1/255.0, bias=[0,0,0])
    class_labels = [f"class_{i}" for i in range(NUM_CLASSES)]
    
    mlmodel = ct.convert(
        traced_model,
        inputs=[image_input],
        outputs=[ct.TensorType(name="classLabelProbs")],
        classifier_config=ct.ClassifierConfig(class_labels=class_labels),
        minimum_deployment_target=ct.target.iOS16
    )
    
    save_path = f"{OUTPUT_NAME}.mlpackage"
    mlmodel.save(save_path)
    print(f"✅ SUKSES! Model CBAM tersimpan di: {save_path}")

if __name__ == "__main__":
    convert_cbam()