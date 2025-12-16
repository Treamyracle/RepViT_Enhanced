import coremltools as ct
import PIL.Image
import time
import numpy as np
import os

# --- KONFIGURASI ---
# Sesuaikan nama file jika berbeda
PATH_SE = "RepViT_SE.mlpackage"
PATH_CBAM = "RepViT_CBAM.mlpackage"
INPUT_SIZE = 224
LOOP_COUNT = 100  # Jumlah iterasi untuk rata-rata

def run_benchmark(model_path, model_label):
    if not os.path.exists(model_path):
        print(f"❌ Error: File {model_path} tidak ditemukan.")
        return None

    print(f"🔄 Memuat model: {model_label}...")
    
    # Load Model dengan konfigurasi 'ALL' (CPU + GPU + Neural Engine)
    config = ct.ComputeUnit.ALL
    try:
        model = ct.models.MLModel(model_path, compute_units=config)
    except Exception as e:
        print(f"Gagal memuat model: {e}")
        return None

    # Buat Dummy Input (Gambar Hitam)
    # CoreML mengharapkan input PIL Image karena kita convert pakai ct.ImageType
    dummy_img = PIL.Image.new('RGB', (INPUT_SIZE, INPUT_SIZE), color='black')
    
    # Nama input di model (biasanya 'image' sesuai script convert sebelumnya)
    # Kita cek input description untuk memastikan
    input_name = model.input_description._fd_spec[0].name
    input_data = {input_name: dummy_img}

    # --- WARM UP ---
    # Penting: Neural Engine butuh waktu untuk 'bangun' dan load model ke memori
    print(f"🔥 Warming up {model_label} (20 iterasi)...")
    for _ in range(20):
        _ = model.predict(input_data)

    # --- BENCHMARK ---
    print(f"🚀 Running Benchmark ({LOOP_COUNT} iterasi)...")
    
    latencies = []
    
    start_total = time.perf_counter()
    for _ in range(LOOP_COUNT):
        start_step = time.perf_counter()
        _ = model.predict(input_data)
        end_step = time.perf_counter()
        latencies.append((end_step - start_step) * 1000) # Convert ke ms
    
    end_total = time.perf_counter()

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95) # 95th percentile (waktu terburuk)
    throughput = LOOP_COUNT / (end_total - start_total)

    print(f"✅ Selesai.\n")
    
    return {
        "avg": avg_latency,
        "p95": p95_latency,
        "fps": throughput
    }

# --- EKSEKUSI ---
print("="*50)
print(f"BENCHMARK MACBOOK M2 (Core ML Native)")
print("="*50)

results_se = run_benchmark(PATH_SE, "RepViT SE (Default)")
results_cbam = run_benchmark(PATH_CBAM, "RepViT CBAM (Ours)")

# --- TAMPILKAN HASIL ---
if results_se and results_cbam:
    print("="*50)
    print(f"{'METRIC':<20} | {'SE (DEFAULT)':<15} | {'CBAM (OURS)':<15}")
    print("-" * 56)
    
    print(f"{'Avg Latency':<20} | {results_se['avg']:.4f} ms      | {results_cbam['avg']:.4f} ms")
    print(f"{'P95 Latency':<20} | {results_se['p95']:.4f} ms      | {results_cbam['p95']:.4f} ms")
    print(f"{'Throughput':<20} | {results_se['fps']:.1f} img/s    | {results_cbam['fps']:.1f} img/s")
    print("-" * 56)
    
    diff = results_cbam['avg'] - results_se['avg']
    status = "LEBIH LAMBAT" if diff > 0 else "LEBIH CEPAT"
    print(f"\nKesimpulan: CBAM {abs(diff):.4f} ms {status} daripada SE.")
    print("="*50)