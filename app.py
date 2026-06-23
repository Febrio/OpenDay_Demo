#!/usr/bin/env python3
"""
CLAUDE GENERATED CODE - Then manually validated by Feb

VLM Object Detection & Tracking Demo
Grounding DINO / Florence-2 + SAM2 — Gradio interface
"""

import os
import sys
import cv2
import torch
import numpy as np
import gradio as gr
import supervision as sv
import tempfile
import shutil
import subprocess
from pathlib import Path
from PIL import Image

# ── Path setup ──────────────────────────────────────────────────────────────
GROUNDED_SAM2_DIR = os.path.expanduser("~/Documents/Grounded-SAM-2")
SAMPLE_VID_DIR = os.path.expanduser("~/Documents/OpenDay_Demo/")
os.chdir(GROUNDED_SAM2_DIR)
sys.path.insert(0, GROUNDED_SAM2_DIR)

SAM2_CHECKPOINT = os.path.join(GROUNDED_SAM2_DIR, "checkpoints/sam2.1_hiera_large.pt")
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_l.yaml"

PRESET_VIDEOS = {
    "🛹 Skateboard": os.path.join(SAMPLE_VID_DIR, "sample_videos/skateboard.mp4"),
    "🐦 Bird":  os.path.join(SAMPLE_VID_DIR, "sample_videos/Bird.mp4"),
    "🦓  Zebra":         os.path.join(SAMPLE_VID_DIR, "sample_videos/zebra.mp4"),
    "🦖 Pocket Monsters":  os.path.join(SAMPLE_VID_DIR, "sample_videos/Pocket_Monsters.mp4"),
    "🧑‍💼 Office work": os.path.join(SAMPLE_VID_DIR, "sample_videos/Office_work.mp4"),
}
PRESET_DEFAULTS = {
    "🛹  Skateboard"    : "skateboard",
    "🐦  Bird"          : "bird",
    "🦓  Zebra"         : "zebra",
    "🦖 Pocket Monsters": "Pokemon",
    "🧑‍💼 Office work" : "office",
}

# ── Global model cache (lazy-loaded on first use) ────────────────────────────
_models: dict = {}
device = "cuda" if torch.cuda.is_available() else "cpu"


def _setup_cuda():
    if torch.cuda.is_available():
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        props = torch.cuda.get_device_properties(0)
        if props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


def _load_sam2():
    if "sam2_video" in _models:
        return
    from sam2.build_sam import build_sam2_video_predictor, build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    _models["sam2_video"]  = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT)
    _models["sam2_img"]    = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT)
    _models["sam2_imgpred"]= SAM2ImagePredictor(_models["sam2_img"])


def _load_grounding_dino():
    if "gdino" in _models:
        return
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    mid = "IDEA-Research/grounding-dino-tiny"
    _models["gdino_proc"]  = AutoProcessor.from_pretrained(mid)
    _models["gdino"]       = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(device)


def _load_florence2():
    if "florence2" in _models:
        return
    from transformers import AutoProcessor, AutoModelForCausalLM
    mid = "microsoft/Florence-2-large"
    _models["florence2"]      = AutoModelForCausalLM.from_pretrained(
        mid, trust_remote_code=True, torch_dtype="auto"
    ).eval().to(device)
    _models["florence2_proc"] = AutoProcessor.from_pretrained(mid, trust_remote_code=True)


# ── Detectors ────────────────────────────────────────────────────────────────

def _detect_gdino(image: Image.Image, text: str, threshold: float):
    _load_grounding_dino()
    prompt = text.lower().strip().rstrip(".") + "."
    inputs = _models["gdino_proc"](images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = _models["gdino"](**inputs)
    results = _models["gdino_proc"].post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        box_threshold=threshold, text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )
    boxes  = results[0]["boxes"].cpu().numpy()
    labels = results[0]["labels"]
    return boxes, list(labels)


def _detect_florence2(image: Image.Image, text: str):
    _load_florence2()
    task   = "<OPEN_VOCABULARY_DETECTION>"
    inputs = _models["florence2_proc"](
        text=task + text, images=image, return_tensors="pt"
    ).to(device, torch.float16)
    gen_ids = _models["florence2"].generate(
        input_ids=inputs["input_ids"].to(device),
        pixel_values=inputs["pixel_values"].to(device),
        max_new_tokens=1024, do_sample=False, num_beams=3,
    )
    gen_text = _models["florence2_proc"].batch_decode(gen_ids, skip_special_tokens=False)[0]
    parsed   = _models["florence2_proc"].post_process_generation(
        gen_text, task=task, image_size=(image.width, image.height)
    )
    result = parsed[task]
    boxes  = np.array(result.get("bboxes", []))
    labels = result.get("labels") or result.get("bboxes_labels", [])
    return boxes, list(labels)


# ── Re-encode for browser compatibility ──────────────────────────────────────

def _reencode_h264(src: str) -> str:
    dst = src.replace(".mp4", "_h264.mp4")
    ret = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-vcodec", "libx264", "-pix_fmt", "yuv420p", dst],
        capture_output=True
    )
    if ret.returncode == 0 and os.path.exists(dst):
        os.remove(src)
        return dst
    return src  # ffmpeg failed — return original


# ── Live webcam per-frame detection ──────────────────────────────────────────

def process_frame_live(frame, text_prompt, confidence):
    """Called on every webcam frame. Returns (annotated_frame, detection_text)."""
    if frame is None:
        return None, ""
    if not text_prompt.strip():
        return frame, "Enter a prompt above to start detecting."

    try:
        # Gradio delivers webcam frames as RGB numpy arrays
        image_pil = Image.fromarray(frame)
        boxes, labels = _detect_gdino(image_pil, text_prompt, confidence)

        if len(boxes) == 0:
            return frame, f"Nothing found for '{text_prompt}' — try a lower threshold."

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        class_ids = np.arange(len(boxes), dtype=np.int32)
        dets = sv.Detections(xyxy=boxes, class_id=class_ids)

        annotated = sv.BoxAnnotator().annotate(scene=frame_bgr.copy(), detections=dets)
        annotated = sv.LabelAnnotator().annotate(annotated, detections=dets, labels=labels)

        info = f"Detected: {', '.join(dict.fromkeys(labels))}  ({len(boxes)} object(s))"
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), info

    except Exception as exc:
        return frame, f"Error: {exc}"


# ── Main processing pipeline ─────────────────────────────────────────────────

def process_video(preset_name, custom_video, text_prompt, detector_choice,
                  confidence, progress=gr.Progress(track_tqdm=True)):

    video_path = custom_video if custom_video else PRESET_VIDEOS.get(preset_name)

    if not video_path or not os.path.exists(video_path):
        return None, "No video selected. Pick a preset or upload a file."
    if not text_prompt.strip():
        return None, "Enter what you want to detect (e.g. 'car', 'hippopotamus')."

    _setup_cuda()

    tmp_frames  = tempfile.mkdtemp(prefix="gsam2_frames_")
    tmp_results = tempfile.mkdtemp(prefix="gsam2_annot_")
    out_path    = tempfile.mktemp(suffix="_out.mp4")

    try:
        # ── 1. Extract frames ──────────────────────────────────────────────
        progress(0.05, desc="Extracting frames…")
        video_info = sv.VideoInfo.from_video_path(video_path)
        with sv.ImageSink(
            target_dir_path=Path(tmp_frames),
            overwrite=True,
            image_name_pattern="{:05d}.jpg",
        ) as sink:
            for frame in sv.get_video_frames_generator(video_path):
                sink.save_image(frame)

        frame_names = sorted(
            [f for f in os.listdir(tmp_frames)
             if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg")],
            key=lambda p: int(os.path.splitext(p)[0]),
        )
        if not frame_names:
            return None, "Could not extract frames from the video."

        # ── 2. Detect on first 5 frames, seed from the best one ──────────
        progress(0.20, desc=f"Running {detector_choice} on candidate frames…")
        seed_frames = min(5, len(frame_names))
        best_boxes, best_labels, best_frame_idx = [], [], 0
        for fi in range(seed_frames):
            img = Image.open(os.path.join(tmp_frames, frame_names[fi])).convert("RGB")
            if detector_choice == "Grounding DINO":
                b, l = _detect_gdino(img, text_prompt, confidence)
            else:
                b, l = _detect_florence2(img, text_prompt)
            if len(b) > len(best_boxes):
                best_boxes, best_labels, best_frame_idx = b, l, fi

        boxes, labels = best_boxes, best_labels
        if len(boxes) == 0:
            hint = "Try lowering the confidence threshold." if detector_choice == "Grounding DINO" else ""
            return None, f"Nothing detected for '{text_prompt}' in the first {seed_frames} frames. {hint}"

        status = f"Found {len(boxes)} object(s) on frame {best_frame_idx}: {', '.join(dict.fromkeys(labels))}"

        # ── 3. Load SAM2 & seed with detected boxes ───────────────────────
        progress(0.35, desc="Loading SAM2 & seeding detections…")
        _load_sam2()
        inference_state = _models["sam2_video"].init_state(video_path=tmp_frames)
        _models["sam2_video"].reset_state(inference_state)

        for obj_id, (_label, box) in enumerate(zip(labels, boxes), start=1):
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            _models["sam2_video"].add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=best_frame_idx,
                obj_id=obj_id,
                points=np.array([[cx, cy]]), labels=np.array([1]),
                box=box,
            )

        # ── 4. Propagate through video ────────────────────────────────────
        progress(0.45, desc="Propagating segmentation through video…")
        video_segments: dict = {}
        for out_idx, out_obj_ids, out_logits in _models["sam2_video"].propagate_in_video(inference_state):
            video_segments[out_idx] = {
                oid: (out_logits[i] > 0.0).cpu().numpy()
                for i, oid in enumerate(out_obj_ids)
            }

        # ── 5. Annotate frames ────────────────────────────────────────────
        progress(0.65, desc="Annotating frames…")
        id_to_label    = {i: lbl for i, lbl in enumerate(labels, start=1)}
        box_ann        = sv.BoxAnnotator()
        label_ann      = sv.LabelAnnotator()
        mask_ann       = sv.MaskAnnotator()
        n_frames       = len(video_segments)

        for frame_idx, segments in video_segments.items():
            img      = cv2.imread(os.path.join(tmp_frames, frame_names[frame_idx]))
            obj_ids  = list(segments.keys())
            seg_masks = np.concatenate(list(segments.values()), axis=0)

            dets = sv.Detections(
                xyxy=sv.mask_to_xyxy(seg_masks),
                mask=seg_masks,
                class_id=np.array(obj_ids, dtype=np.int32),
            )
            frame_labels = [id_to_label.get(i, "object") for i in obj_ids]

            ann = box_ann.annotate(scene=img.copy(), detections=dets)
            ann = label_ann.annotate(ann, detections=dets, labels=frame_labels)
            ann = mask_ann.annotate(scene=ann, detections=dets)
            cv2.imwrite(os.path.join(tmp_results, f"{frame_idx:05d}.jpg"), ann)

            if frame_idx % max(1, n_frames // 20) == 0:
                p = 0.65 + 0.25 * frame_idx / max(n_frames, 1)
                progress(p, desc=f"Annotating frame {frame_idx + 1}/{n_frames}…")

        # ── 6. Assemble output video ──────────────────────────────────────
        progress(0.92, desc="Assembling output video…")
        from utils.video_utils import create_video_from_images
        create_video_from_images(tmp_results, out_path, frame_rate=int(video_info.fps or 25))

        # Re-encode to H.264 for browser playback
        out_path = _reencode_h264(out_path)

        progress(1.0, desc="Done!")
        return out_path, f"Done! {status}"

    except Exception as exc:
        import traceback
        return None, f"Error: {exc}\n{traceback.format_exc()}"

    finally:
        shutil.rmtree(tmp_frames,  ignore_errors=True)
        shutil.rmtree(tmp_results, ignore_errors=True)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
#title  { text-align: center; }
#run-btn { font-size: 1.1rem; }
"""

def build_ui():
    with gr.Blocks(title="VLM Object Detection Demo") as demo:

        gr.Markdown(
            "# Video Object Detection & Tracking Demo\n"
            "Open-vocabulary detection powered by **Grounding DINO** / **Florence-2** + **SAM2**",
            elem_id="title",
        )

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════
            # Tab 1 — Video Detection & Tracking
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("📹  Video Detection"):
                gr.Markdown(
                    "Detect and **track** any object through a full video clip. "
                    "Grounding DINO finds it on frame 0 — SAM2 follows it to the end."
                )
                with gr.Row():
                    # Left: controls
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("### Video")
                        preset_dd = gr.Radio(
                            choices=list(PRESET_VIDEOS.keys()),
                            value="🛹 Skateboard",
                            label="Preset video",
                        )
                        custom_vid = gr.Video(
                            label="Or upload your own video", sources=["upload"]
                        )

                        gr.Markdown("### Detection")
                        text_box = gr.Textbox(
                            label="What to detect",
                            placeholder="e.g.  hippopotamus   /   car   /   person with backpack",
                            value="hippopotamus",
                        )
                        detector_radio = gr.Radio(
                            choices=["Grounding DINO", "Florence-2"],
                            value="Grounding DINO",
                            label="Detection model",
                            info="Grounding DINO is faster. Florence-2 handles richer descriptions.",
                        )
                        conf_slider = gr.Slider(
                            minimum=0.10, maximum=0.90, value=0.35, step=0.05,
                            label="Confidence threshold  (Grounding DINO)",
                        )
                        run_btn = gr.Button(
                            "Run Detection & Tracking", variant="primary",
                            size="lg", elem_id="run-btn",
                        )

                    # Right: video output
                    with gr.Column(scale=2):
                        gr.Markdown("### Preview")
                        input_display = gr.Video(label="Input video", interactive=False)
                        gr.Markdown("### Result")
                        output_video = gr.Video(
                            label="Detected & tracked objects", interactive=False
                        )
                        status_box = gr.Textbox(
                            label="Status", interactive=False, lines=2
                        )

                # Events
                def on_preset(name):
                    return PRESET_VIDEOS.get(name, None), PRESET_DEFAULTS.get(name, "")

                preset_dd.change(
                    fn=on_preset,
                    inputs=[preset_dd],
                    outputs=[input_display, text_box],
                )

                def on_upload(vid):
                    return vid, gr.update(value=None)

                custom_vid.change(
                    fn=on_upload,
                    inputs=[custom_vid],
                    outputs=[input_display, preset_dd],
                )

                run_btn.click(
                    fn=process_video,
                    inputs=[preset_dd, custom_vid, text_box, detector_radio, conf_slider],
                    outputs=[output_video, status_box],
                )

                demo.load(
                    fn=lambda: (PRESET_VIDEOS["🛹 Skateboard"], "skateboard"),
                    outputs=[input_display, text_box],
                )

            # ══════════════════════════════════════════════════════════════
            # Tab 2 — Live Webcam
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("📷  Live Webcam"):
                gr.Markdown(
                    "Point your camera at **anything** and type what you want the AI to find. "
                    "Detection runs on the server GPU — your device just sends frames."
                )

                with gr.Row():
                    # Left: controls
                    with gr.Column(scale=1, min_width=260):
                        live_prompt = gr.Textbox(
                            label="What to detect",
                            placeholder="e.g.  person   /   bottle   /   laptop",
                            value="person",
                        )
                        live_conf = gr.Slider(
                            minimum=0.10, maximum=0.90, value=0.30, step=0.05,
                            label="Confidence threshold",
                        )
                        gr.Markdown(
                            "_Change the prompt or threshold at any time — "
                            "the next frame will use the new value instantly._\n\n"
                            "**Tip:** keep prompts short and specific for best results."
                        )

                    # Right: webcam + output
                    with gr.Column(scale=2):
                        with gr.Row():
                            webcam_in = gr.Image(
                                sources=["webcam"],
                                streaming=True,
                                type="numpy",       # Gradio 6 requires this to be explicit
                                label="Your camera",
                                height=400,
                            )
                            webcam_out = gr.Image(
                                label="AI detection",
                                interactive=False,
                                height=400,
                            )
                        detect_log = gr.Textbox(
                            label="Detection log",
                            interactive=False,
                            lines=1,
                        )

                # Stream every frame to the detection function
                webcam_in.stream(
                    fn=process_frame_live,
                    inputs=[webcam_in, live_prompt, live_conf],
                    outputs=[webcam_out, detect_log],
                    stream_every=0.1,   # max 10 fps; GPU inference is the real bottleneck
                    concurrency_limit=1,
                )

    return demo


if __name__ == "__main__":
    _setup_cuda()
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=8000,
        share=True,   # creates a public HTTPS tunnel — webcam requires HTTPS
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS,
        allowed_paths=[os.path.join(SAMPLE_VID_DIR, "sample_videos")],
    )
