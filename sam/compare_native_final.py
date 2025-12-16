import coremltools as ct
import numpy as np
import time
import os

def benchmark_coreml_model(model_path, name, input_shape):
    print(f"\n--- Menguji {name} ---")
    if not os.path.exists(model_path):
        print(f"❌ File {model_path} tidak ditemukan.")
        return None

    try:
        # Load Model
        # compute_units=ct.ComputeUnit.ALL akan otomatis memilih Neural Engine/GPU
        model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.ALL)
        
        # Buat Dummy Input
        # CoreML input biasanya dictionary. Key input default biasanya 'image' atau 'x_1'
        # Kita cek nama input pertama dari metadata model
        input_desc = model.get_spec().description.input[0]
        input_name = input_desc.name
        
        # Buat random data numpy
        # Shape di CoreML biasanya (1, 3, 1024, 1024)
        dummy_data = np.random.rand(*input_shape).astype(np.float32)
        
        input_dict = {input_name: dummy_data}
        
        # Warmup (Penting!)
        print("Warming up...")
        for _ in range(3):
            model.predict(input_dict)
            
        # Benchmark Loop
        print("Running benchmark (10 loops)...")
        latencies = []
        for _ in range(10):
            start = time.time()
            model.predict(input_dict)
            end = time.time()
            latencies.append((end - start) * 1000) # ke ms

        avg_latency = sum(latencies) / len(latencies)
        print(f"✅ Rata-rata Latency: {avg_latency:.2f} ms")
        return avg_latency

    except Exception as e:
        print(f"Error testing {name}: {e}")
        return None

def main():
    print("=== NATIVE COREML BENCHMARK (Mac/ANE) ===\n")

    # 1. Test RepViT Encoder
    # Input RepViT biasanya perlu shape spesifik sesuai export sebelumnya
    t_repvit = benchmark_coreml_model(
        "coreml/repvit_1024.mlpackage", 
        "RepViT-SAM Encoder", 
        (1, 3, 1024, 1024)
    )

    # 2. Test MobileSAM Encoder
    t_mobile = benchmark_coreml_model(
        "coreml/mobilesam_encoder.mlpackage", 
        "MobileSAM Encoder", 
        (1, 3, 1024, 1024)
    )

    # Kesimpulan
    print("\n" + "="*30)
    print("       HASIL AKHIR       ")
    print("="*30)
    
    if t_repvit and t_mobile:
        print(f"RepViT-SAM : {t_repvit:.2f} ms")
        print(f"MobileSAM  : {t_mobile:.2f} ms")
        
        diff = t_mobile - t_repvit
        if diff > 0:
            speedup = t_mobile / t_repvit
            print(f"\n🏆 RepViT LEBIH CEPAT {speedup:.2f}x lipat!")
            print("(Ini membuktikan RepViT lebih optimal di Neural Engine)")
        else:
            print(f"\nMobileSAM lebih cepat. (Mungkin RepViT belum ter-fuse sempurna atau fallback ke CPU)")
    else:
        print("Gagal menjalankan salah satu model.")

if __name__ == "__main__":
    main()