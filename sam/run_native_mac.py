import torch
import numpy as np
import cv2
import coremltools as ct
import matplotlib.pyplot as plt
import time
from repvit_sam.utils.transforms import ResizeLongestSide
import torch.nn.functional as F

# --- FUNGSI BANTUAN (Dari coreml_example.ipynb) ---
def preprocess(x, img_size=1024):
    transform = ResizeLongestSide(img_size)
    x = transform.apply_image(x)
    x = torch.as_tensor(x).permute(2, 0, 1).contiguous()
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x, transform

def show_result(image, mask, points, labels):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Overlay Mask
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.imshow(mask_image)
    
    # Plot Points
    pos_points = points[labels==1]
    neg_points = points[labels==0]
    plt.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=375, edgecolor='white', linewidth=1.25)
    plt.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=375, edgecolor='white', linewidth=1.25)
    plt.axis('off')
    plt.show()

# --- MAIN PROGRAM ---
def main():
    print("Loading CoreML Models (Running on Apple Neural Engine)...")
    
    # Load Models
    # 'compute_units' menentukan hardware: .all (Neural Engine + GPU + CPU)
    encoder = ct.models.MLModel('coreml/repvit_1024.mlpackage')
    decoder = ct.models.MLModel('coreml/sam_decoder.mlpackage')
    
    # Load Image
    img_path = 'app/assets/picture1.jpg' # Pastikan path gambar benar
    raw_image = cv2.imread(img_path)
    if raw_image is None:
        print("Error: Gambar tidak ditemukan.")
        return
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    
    print("Processing Image...")
    # Preprocess
    input_image, transform = preprocess(raw_image)
    
    # 1. Jalankan ENCODER
    # Di Mac, ini akan sangat cepat karena menggunakan hardware akselerasi
    start_enc = time.time()
    # CoreML input dictionary keys harus sesuai dengan nama input saat export
    encoder_input = {'x_1': input_image.numpy()[None, ...]} 
    encoder_output = encoder.predict(encoder_input)
    image_embedding = list(encoder_output.values())[0]
    end_enc = time.time()
    
    print(f"Encoder Latency: {(end_enc - start_enc) * 1000:.2f} ms")

    # 2. Siapkan Prompt (Titik Tengah)
    h, w = raw_image.shape[:2]
    
    # Koordinat Asli (1 titik)
    input_point = np.array([[w//2, h//2]]) # Titik tengah
    input_label = np.array([1])            # Label 1 (Foreground)

    # Transformasi koordinat asli dulu ke ukuran 1024x1024
    coreml_coord = input_point[None, :, :].astype(np.float32)
    coreml_coord = transform.apply_coords(coreml_coord, raw_image.shape[:2]).astype(np.float32)
    
    # === PERBAIKAN: PADDING MENJADI 5 TITIK ===
    # Model CoreML mengharapkan input fixed (1, 5, 2) untuk coords dan (1, 5) untuk labels.
    # Kita akan isi 4 titik sisanya dengan 0 (padding).
    
    # 1. Pad Coordinates: (1, 1, 2) -> (1, 5, 2)
    pad_coord = np.zeros((1, 4, 2), dtype=np.float32)
    coreml_coord_padded = np.concatenate([coreml_coord, pad_coord], axis=1)

    # 2. Pad Labels: (1, 1) -> (1, 5)
    # Gunakan label -1 (biasanya berarti ignore) atau 0 (background) untuk padding
    coreml_label = input_label[None, :].astype(np.float32)
    pad_label = -1 * np.ones((1, 4), dtype=np.float32) 
    coreml_label_padded = np.concatenate([coreml_label, pad_label], axis=1)
    
    # ==========================================

    # Input Decoder lainnya (Mask kosong)
    coreml_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    coreml_has_mask_input = np.zeros(1, dtype=np.float32)

    # 3. Jalankan DECODER (Gunakan input yang sudah di-padding)
    decoder_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": coreml_coord_padded,  # <--- Ganti ini
        "point_labels": coreml_label_padded,  # <--- Ganti ini
        "mask_input": coreml_mask_input,
        "has_mask_input": coreml_has_mask_input,
    }
    
    start_dec = time.time()
    decoder_output = decoder.predict(decoder_inputs)
    end_dec = time.time()
    
    # ... (kode sebelumnya tetap sama sampai baris print Latency) ...
    
    print(f"Decoder Latency: {(end_dec - start_dec) * 1000:.2f} ms")
    
    # === PERBAIKAN: DETEKSI OTOMATIS NAMA OUTPUT ===
    # Print semua key yang tersedia untuk debugging
    print(f"Output Keys yang tersedia: {list(decoder_output.keys())}")

    masks = None
    
    # Coba cari key yang mengandung kata 'mask'
    if 'masks' in decoder_output:
        masks = decoder_output['masks']
    elif 'low_res_masks' in decoder_output:
        masks = decoder_output['low_res_masks']
    else:
        # Fallback: Jika nama aneh (misal var_123), ambil output pertama
        # Biasanya output SAM urutannya: masks, iou_pred, low_res_masks
        # Kita asumsi output pertama yang dimensi-nya paling besar adalah mask
        for key, val in decoder_output.items():
            print(f"Cek output '{key}' shape: {val.shape}")
            # Mask biasanya 4 dimensi (1, 1, 256, 256) atau (1, 4, 256, 256)
            if len(val.shape) == 4 and val.shape[-1] == 256:
                print(f"-> Menggunakan '{key}' sebagai mask.")
                masks = val
                break
        
        # Jika masih belum ketemu, ambil key pertama saja secara paksa
        if masks is None:
            first_key = list(decoder_output.keys())[0]
            print(f"Warning: Mengambil output pertama '{first_key}' secara paksa.")
            masks = decoder_output[first_key]
    # ===============================================
    
    # Postprocess untuk visualisasi
    # Convert ke Torch Tensor untuk resize
    mask_tensor = torch.tensor(masks)
    
    # RepViT-SAM decoder output biasanya low-res (256x256), perlu di-upscale ke (1024x1024) atau ukuran asli
    # Kita resize ke ukuran asli gambar (h, w)
    mask_resized = F.interpolate(mask_tensor, size=(h, w), mode="bilinear", align_corners=False)
    
    # Thresholding (Logits -> Binary Mask)
    # Biasanya mask terbaik ada di index pertama [0, 0] setelah padding
    mask_final = (mask_resized > 0.0).numpy()[0, 0]

    print("Menampilkan hasil...")
    show_result(raw_image, mask_final, input_point, input_label)

if __name__ == "__main__":
    main()