import os
import time
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
from repvit_sam import SamPredictor as RepVitPredictor, sam_model_registry as repvit_registry

# --- IMPORT MOBILESAM ---
try:
    from mobile_sam import SamPredictor as MobilePredictor, sam_model_registry as mobile_registry
    MOBILE_SAM_AVAILABLE = True
except ImportError:
    print("MobileSAM tidak ditemukan.")
    MOBILE_SAM_AVAILABLE = False

from utils.tools import format_results, point_prompt
from utils.tools_gradio import fast_process

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ==========================================
# 1. SETUP MODEL
# ==========================================

# RepViT-SAM
print("Loading RepViT-SAM...")
repvit_checkpoint = "../weights/repvit_sam.pt"
repvit_model = repvit_registry["repvit"](checkpoint=repvit_checkpoint)
repvit_model.to(device=device)
repvit_model.eval()
if hasattr(repvit_model.image_encoder, 'fuse'):
    repvit_model.image_encoder.fuse()
predictor_repvit = RepVitPredictor(repvit_model)

# MobileSAM
predictor_mobile = None
if MOBILE_SAM_AVAILABLE:
    print("Loading MobileSAM...")
    mobile_checkpoint = "../weights/mobile_sam.pt"
    if os.path.exists(mobile_checkpoint):
        mobile_model = mobile_registry["vit_t"](checkpoint=mobile_checkpoint)
        mobile_model.to(device=device)
        mobile_model.eval()
        predictor_mobile = MobilePredictor(mobile_model)

# ==========================================
# 2. FUNGSI BENCHMARK OTOMATIS
# ==========================================
def run_benchmark_test():
    # Daftar 6 gambar aset
    image_files = [f"app/assets/picture{i}.jpg" for i in range(1, 7)]
    valid_images = [f for f in image_files if os.path.exists(f)]
    
    if not valid_images:
        return "Error: Tidak ditemukan gambar di app/assets/picture1.jpg s/d picture6.jpg"

    output_log = f"### Benchmark Results ({len(valid_images)} Images)\n"
    output_log += "Device: " + str(device) + "\n\n"

    # Daftar model yang akan dites
    models_to_test = [("RepViT-SAM", predictor_repvit)]
    if predictor_mobile:
        models_to_test.append(("MobileSAM", predictor_mobile))

    for model_name, predictor in models_to_test:
        latencies = []
        print(f"Benchmarking {model_name}...")
        
        # Warmup (Penting agar run pertama tidak bias)
        dummy_img = np.array(Image.open(valid_images[0]).convert('RGB'))
        predictor.set_image(dummy_img)

        for img_path in valid_images:
            # Load Image
            image = Image.open(img_path).convert('RGB')
            nd_image = np.array(image)
            
            # Titik Prompt (Tengah Gambar)
            h, w = nd_image.shape[:2]
            input_point = np.array([[w//2, h//2]])
            input_label = np.array([1])

            # Sinkronisasi Waktu (untuk GPU)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            start_time = time.time()

            # Proses Utama: Encoder + Decoder
            predictor.set_image(nd_image)
            predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False
            )

            if torch.cuda.is_available(): torch.cuda.synchronize()
            end_time = time.time()

            # Hitung durasi (ms)
            latencies.append((end_time - start_time) * 1000)
        
        avg_time = sum(latencies) / len(latencies)
        output_log += f"**{model_name}**:\n"
        output_log += f"- Avg Latency: **{avg_time:.2f} ms**\n"
        output_log += f"- Min: {min(latencies):.2f} ms | Max: {max(latencies):.2f} ms\n\n"
    
    # Hitung Speedup jika ada kedua model
    if len(models_to_test) == 2:
        repvit_time = [l for m, l in models_to_test if m == "RepViT-SAM"][0]
        # Kita butuh nilai rata-rata tadi, ambil dari log parsing simple atau hitung ulang
        # (Untuk simplifikasi kode UI, kita tampilkan teks saja)
        pass

    return output_log

# ==========================================
# 3. FUNGSI SEGMENTASI MANUAL (INTERAKTIF)
# ==========================================
global_points = []
global_point_label = []

def segment_with_points(image, original_image, model_choice):
    global global_points, global_point_label

    if model_choice == "MobileSAM" and predictor_mobile:
        active_predictor = predictor_mobile
    else:
        active_predictor = predictor_repvit

    w, h = image.size
    input_size = 1024
    scale = input_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    image_resized = image.resize((new_w, new_h))
    scaled_points = np.array([[int(x * scale) for x in p] for p in global_points])
    scaled_labels = np.array(global_point_label)

    if scaled_points.size == 0: return image_resized, image

    nd_image = np.array(original_image.resize((new_w, new_h)))
    active_predictor.set_image(nd_image)
    masks, scores, logits = active_predictor.predict(
        point_coords=scaled_points,
        point_labels=scaled_labels,
        multimask_output=False,
    )

    results = format_results(masks, scores, logits, 0)
    annotations, _ = point_prompt(results, scaled_points, scaled_labels, new_h, new_w)
    annotations = np.array([annotations])

    fig = fast_process(annotations=annotations, image=image_resized, device=device, scale=(1024 // input_size))
    
    global_points = []
    global_point_label = []
    return fig, original_image.resize((new_w, new_h))

def get_points_with_draw(image, label, evt: gr.SelectData):
    global global_points, global_point_label
    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 15 * (max(image.size)/1024), (255, 255, 0) if label == "Add Mask" else (255, 0, 255)
    global_points.append([x, y])
    global_point_label.append(1 if label == "Add Mask" else 0)
    
    draw = ImageDraw.Draw(image)
    draw.ellipse([(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)], fill=point_color)
    return image

def clear_data():
    global global_points, global_point_label
    global_points = []
    global_point_label = []
    return None, None

# ==========================================
# 4. MEMBANGUN UI
# ==========================================
css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"
title = "<center><strong><font size='8'>RepViT-SAM Playground</font></strong></center>"

examples = [[f"app/assets/picture{i}.jpg"] for i in range(1, 5) if os.path.exists(f"app/assets/picture{i}.jpg")]
default_img = examples[0][0] if examples else None

with gr.Blocks(css=css, title="RepViT-SAM") as demo:
    gr.Markdown(title)
    
    # --- TAB 1: INTERACTIVE MODE ---
    with gr.Tab("Interactive Mode"):
        original_image = gr.State(value=Image.open(default_img).convert('RGB') if default_img else None)
        with gr.Row():
            with gr.Column():
                cond_img_p = gr.Image(label="Input", value=default_img, type="pil", interactive=True)
                model_selector = gr.Radio(["RepViT-SAM", "MobileSAM"], label="Model", value="RepViT-SAM")
                point_type = gr.Radio(["Add Mask", "Remove Area"], label="Point Type", value="Add Mask")
                with gr.Row():
                    seg_btn = gr.Button("Segment", variant="primary")
                    restart_btn = gr.Button("Clear", variant="secondary")
            with gr.Column():
                res_img = gr.Image(label="Result", type="pil", interactive=False)
        
        # Events
        cond_img_p.select(get_points_with_draw, [cond_img_p, point_type], cond_img_p)
        cond_img_p.upload(lambda x: x, inputs=[cond_img_p], outputs=[original_image])
        seg_btn.click(segment_with_points, [cond_img_p, original_image, model_selector], [res_img, cond_img_p])
        restart_btn.click(clear_data, outputs=[cond_img_p, res_img])
        gr.Examples(examples, inputs=[cond_img_p], outputs=[original_image], fn=lambda x: x, run_on_click=True)

    # --- TAB 2: AUTOMATIC BENCHMARK ---
    with gr.Tab("Benchmark Mode"):
        gr.Markdown("### Automatic Speed Test (Average of 6 Images)")
        gr.Markdown("This will run inference on all images in `app/assets/` using a center point prompt.")
        
        bench_btn = gr.Button("Start Benchmark 🚀", variant="primary")
        bench_output = gr.Markdown("Waiting to start...")
        
        bench_btn.click(run_benchmark_test, outputs=bench_output)

# Launch
demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860)