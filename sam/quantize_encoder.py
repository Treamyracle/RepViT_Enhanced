import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

# Nama file input (yang sudah Anda fuse sebelumnya)
input_model_path = "repvit_sam_image_encoder.onnx"
# Nama file output (hasil optimasi)
output_model_path = "repvit_encoder_quantized.onnx"

print(f"Mengoptimalkan {input_model_path} menjadi INT8...")

try:
    # Perbaikan: Menghapus parameter optimize_model=True yang menyebabkan error
    quantize_dynamic(
        model_input=input_model_path,
        model_output=output_model_path,
        weight_type=QuantType.QUInt8
    )
    print(f"✅ Sukses! Model tersimpan di: {output_model_path}")
    
    # Cek ukuran file
    if os.path.exists(output_model_path):
        size_before = os.path.getsize(input_model_path) / (1024*1024)
        size_after = os.path.getsize(output_model_path) / (1024*1024)
        print(f"📉 Ukuran file berkurang dari {size_before:.2f} MB menjadi {size_after:.2f} MB")
    else:
        print("❌ File output tidak ditemukan.")

except Exception as e:
    print(f"❌ Error: {e}")