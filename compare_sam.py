import torch
import numpy as np
import cv2
import time
import glob
import os
import sys

# Import Library
try:
    from repvit_sam import sam_model_registry as repvit_registry
    from repvit_sam import SamPredictor as RepVitPredictor
except ImportError:
    print("Error: Gagal import repvit_sam.")
    sys.exit(1)

try:
    from mobile_sam import sam_model_registry as mobile_registry
    from mobile_sam import SamPredictor as MobilePredictor
except ImportError:
    print("Error: Gagal import mobile_sam.")
    sys.exit(1)

def setup_model(model_type, checkpoint_path, device, is_repvit=False):
    print(f"Loading {model_type} from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        print(f"Warning: File checkpoint tidak ditemukan di {checkpoint_path}")
        return None

    if is_repvit:
        # RepViT Setup
        sam = repvit_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        sam.eval()
        
        # === LANGKAH KUNCI: RE-PARAMETERIZATION (FUSE) ===
        # Tanpa ini, RepViT berjalan sangat lambat (mode training)
        print(" -> Melakukan structural re-parameterization (fusing)...")
        if hasattr(sam.image_encoder, 'fuse'):
            sam.image_encoder.fuse()
        else:
            print("Warning: Method .fuse() tidak ditemukan pada image_encoder")

        predictor = RepVitPredictor(sam)
    else:
        # MobileSAM Setup
        sam = mobile_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        sam.eval()
        predictor = MobilePredictor(sam)
        
    return predictor

def run_inference_test(predictor, image_paths, model_name, device):
    if predictor is None: return 0
        
    latencies = []
    print(f"\n--- Testing {model_name} ---")
    
    # Warmup GPU
    if len(image_paths) > 0:
        dummy = cv2.imread(image_paths[0])
        if dummy is not None:
            predictor.set_image(dummy)
    
    print(f"Mulai benchmark pada {len(image_paths)} gambar...")
    
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prompt dummy (tengah gambar)
        h, w = image.shape[:2]
        input_point = np.array([[w//2, h//2]])
        input_label = np.array([1])
        
        # Ukur waktu
        if device == 'cuda': torch.cuda.synchronize()
        start_time = time.time()
        
        # Proses Utama: Encoder (berat) + Decoder (ringan)
        predictor.set_image(image) # Bagian ini yang dipercepat oleh fuse()
        predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        
        if device == 'cuda': torch.cuda.synchronize()
        end_time = time.time()
        
        latencies.append((end_time - start_time) * 1000)

    if not latencies: return 0
    avg_latency = sum(latencies) / len(latencies)
    print(f"Rata-rata Latency {model_name}: {avg_latency:.2f} ms")
    return avg_latency

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Cari gambar
    images = glob.glob("imagenette2-320/val/**/*.JPEG", recursive=True)[:50]
    if not images:
        images = glob.glob("**/val/**/*.JPEG", recursive=True)[:50]
    
    if not images:
        print("Dataset tidak ditemukan. Pastikan folder imagenette2-320 ada.")
        return

    # 1. RepViT-SAM (Perhatikan key: 'repvit', bukan 'repvit_sam')
    repvit_predictor = setup_model(
        model_type="repvit", 
        checkpoint_path="weights/repvit_sam.pt", 
        device=device,
        is_repvit=True
    )
    t_repvit = run_inference_test(repvit_predictor, images, "RepViT-SAM", device)

    # 2. MobileSAM
    mobile_predictor = setup_model(
        model_type="vit_t", 
        checkpoint_path="weights/mobile_sam.pt", 
        device=device,
        is_repvit=False
    )
    t_mobile = run_inference_test(mobile_predictor, images, "MobileSAM", device)

    # Hasil
    print("\n=== HASIL AKHIR ===")
    if t_repvit > 0: print(f"RepViT: {t_repvit:.2f} ms")
    if t_mobile > 0: print(f"Mobile : {t_mobile:.2f} ms")

if __name__ == "__main__":
    main()