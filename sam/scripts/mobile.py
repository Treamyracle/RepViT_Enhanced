import torch
import coremltools as ct
import sys
import os
from argparse import ArgumentParser

# Path Setup
sys.path.append(os.getcwd())
try:
    import repvit_sam.modeling
    from repvit_sam.utils.coreml import SamCoreMLModel
    from repvit_sam import sam_model_registry
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    import repvit_sam.modeling
    from repvit_sam.utils.coreml import SamCoreMLModel
    from repvit_sam import sam_model_registry

def run_export():
    checkpoint = "../weights/repvit_sam.pt"
    output_path = "coreml/sam_decoder.mlpackage"
    
    print(f"Loading Decoder from {checkpoint}...")
    # Load Model RepViT (Kita ambil decodernya saja)
    sam = sam_model_registry['repvit'](checkpoint=checkpoint)
    
    # Bungkus dengan CoreML Wrapper
    onnx_model = SamCoreMLModel(
        model=sam,
        orig_img_size=[1024, 1024],
        return_single_mask=True, # Penting agar output mask cuma 1
        use_stability_score=False,
        return_extra_metrics=False # False = Outputnya biasanya: masks, iou_predictions, low_res_masks
    )
    onnx_model.eval()

    # Dummy Inputs
    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = (64, 64)
    mask_input_size = [4 * x for x in embed_size]
    
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size),
        "point_coords": torch.randint(0, 1024, (1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(0, 4, (1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
    }

    print("Tracing...")
    traced_model = torch.jit.trace(onnx_model, example_inputs=list(dummy_inputs.values()))

    print("Converting with Named Outputs...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name='image_embeddings', shape=dummy_inputs['image_embeddings'].shape),
            ct.TensorType(name='point_coords', shape=(1, 5, 2)),
            ct.TensorType(name='point_labels', shape=(1, 5)),
            ct.TensorType(name='mask_input', shape=dummy_inputs['mask_input'].shape),
            ct.TensorType(name='has_mask_input', shape=dummy_inputs['has_mask_input'].shape),
        ],
        # --- BAGIAN PENTING: MENAMAI OUTPUT ---
        outputs=[
            ct.TensorType(name="masks"),
            ct.TensorType(name="iou_predictions"),
            ct.TensorType(name="low_res_masks")
        ],
        # --------------------------------------
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16
    )

    mlmodel.save(output_path)
    print(f"✅ Selesai! Output decoder sekarang bernama 'masks' dan 'iou_predictions'.")

if __name__ == "__main__":
    run_export()