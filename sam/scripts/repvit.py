import coremltools as ct
import coremltools.optimize.coreml as cto

model_path = "coreml/repvit_1024.mlpackage"
output_path = "coreml/repvit_1024_int8.mlpackage"

print(f"Loading {model_path}...")
model = ct.models.MLModel(model_path)

print("Applying Linear 8-bit Quantization...")
# Menggunakan algoritma kuantisasi linear yang ramah Neural Engine
config = cto.OptimizationConfig(
    global_config=cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", # Mode paling stabil untuk ANE
        dtype="int8",
        granularity="per_tensor"
    )
)

compressed_model = cto.linear_quantize_weights(model, config=config)

compressed_model.save(output_path)
print(f"✅ Selesai! Disimpan di {output_path}")