import torch
import coremltools as ct
from model.repvit import repvit_m2_3
import os

# --- KONFIGURASI SE ---
MODEL_PATH = "checkpoint_best_SE.pth" # Ganti dengan path weight SE Anda
NUM_CLASSES = 10
INPUT_SIZE = 224
OUTPUT_NAME = "RepViT_M2_3_SE"

def convert_se():
    print(f"🔄 Memulai Konversi RepViT (SE)...")
    
    # 1. Inisialisasi Model Standar (Default RepViT pake SE)
    model = repvit_m2_3(num_classes=NUM_CLASSES)
    
    # 2. Load Weights
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: File {MODEL_PATH} tidak ditemukan.")
        return

    print(f"📂 Loading weights: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    # Handle dictionary keys
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"⚠️ Warning saat load state_dict: {e}")
        
    model.eval()
    
    # 3. Trace
    print("⚡ Tracing model...")
    example_input = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
    traced_model = torch.jit.trace(model, example_input)
    
    # 4. Convert ke CoreML
    print("📦 Converting ke CoreML...")
    image_input = ct.ImageType(name="image", shape=example_input.shape, scale=1/255.0, bias=[0,0,0])
    class_labels = [f"class_{i}" for i in range(NUM_CLASSES)]
    
    mlmodel = ct.convert(
        traced_model,
        inputs=[image_input],
        outputs=[ct.TensorType(name="classLabelProbs")],
        classifier_config=ct.ClassifierConfig(class_labels=class_labels),
        minimum_deployment_target=ct.target.iOS16
    )
    
    # 5. Simpan
    save_path = f"{OUTPUT_NAME}.mlpackage"
    mlmodel.save(save_path)
    print(f"✅ SUKSES! Model SE tersimpan di: {save_path}")

if __name__ == "__main__":
    convert_se()