import torch
import coremltools as ct
from mobile_sam import sam_model_registry
import os

# 1. Load MobileSAM Model
checkpoint = "../weights/mobile_sam.pt"
if not os.path.exists(checkpoint):
    print(f"Error: {checkpoint} tidak ditemukan.")
    exit()

print("Loading MobileSAM (TinyViT)...")
model = sam_model_registry["vit_t"](checkpoint=checkpoint)
image_encoder = model.image_encoder
image_encoder.eval()

# 2. Siapkan Dummy Input (1024x1024)
# MobileSAM menggunakan input standar (1, 3, 1024, 1024)
dummy_input = torch.randn(1, 3, 1024, 1024)

# 3. Tracing dengan TorchScript
print("Tracing model...")
traced_model = torch.jit.trace(image_encoder, dummy_input)

# 4. Convert ke CoreML
print("Converting to CoreML (FP16)...")
# Kita gunakan FP16 agar adil dengan RepViT (dan agar bisa masuk Neural Engine)
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="image", shape=(1, 3, 1024, 1024))],
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS16 # Sesuaikan dengan OS Anda
)

# 5. Simpan
output_path = "coreml/mobilesam_encoder.mlpackage"
mlmodel.save(output_path)
print(f"✅ Sukses! Tersimpan di {output_path}")