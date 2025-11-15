# main.py
import os
import time
import json
from pathlib import Path
import psutil
import torch

from src.scene_cutting import get_scene_list
from src.frame_sampling import sample_frames
from src.frame_captioning_blip import caption_frames
from src.debug_utils import see_scenes_cuts
from src.audio_utils import extract_scene_audio_ffmpeg, get_ffmpeg_executable
from src.audio_asr import extract_speech_asr
from src.audio_natural import extract_natural_audio_labels

# ----------------------------
# CONFIG / PATHS
# ----------------------------
VIDEO_PATH = r"Videos\Watch Malala Yousafzai's Nobel Peace Prize acceptance speech.mp4"
OUTPUT_DIR = Path("./output")
FRAMES_DIR = OUTPUT_DIR / "frames"
AUDIO_DIR = OUTPUT_DIR / "audio"
CAPTIONS_DIR = OUTPUT_DIR / "captions"

for d in [FRAMES_DIR, AUDIO_DIR, CAPTIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")
FFMPEG_PATH = get_ffmpeg_executable()

# ----------------------------
# SCENE DETECTION
# ----------------------------
print("Detecting scenes...")
scenes = get_scene_list(VIDEO_PATH)
see_scenes_cuts(scenes)

# ----------------------------
# FRAME SAMPLING & BLIP CAPTIONS
# ----------------------------
print("Sampling frames and generating BLIP captions...")
scenes_with_frames = sample_frames(input_video_path=VIDEO_PATH, scenes=scenes, num_frames=2, output_dir=str(FRAMES_DIR))
captioned_scenes = caption_frames(scenes=scenes_with_frames, max_length=30, num_beams=4, do_sample=False, debug=False, prompt="a video frame of")

# Combine BLIP captions per scene
for scene in captioned_scenes:
    frames = scene.get("frames", [])
    frame_captions = scene.get("frame_captions", [])
    frame_paths = scene.get("frame_paths", [])
    scene["frames"] = [
        {"image": f, "path": frame_paths[i] if i < len(frame_paths) else "", "caption": frame_captions[i] if i < len(frame_captions) else ""}
        for i, f in enumerate(frames)
    ]
    scene["blip_caption"] = " ".join([f["caption"] for f in scene["frames"]])

# ----------------------------
# AUDIO EXTRACTION + ASR + AST
# ----------------------------
scene_metrics = []
video_start_time = time.time()

for scene in captioned_scenes:
    start_sec, end_sec = scene["start_seconds"], scene["end_seconds"]
    idx = scene["scene_index"]

    # Extract audio
    audio_path = AUDIO_DIR / f"scene_{idx:02d}.wav"
    extract_scene_audio_ffmpeg(str(VIDEO_PATH), str(audio_path), start_sec, end_sec)
    scene["audio_path"] = str(audio_path)

    # ASR (Whisper with VAD)
    scene["speech"] = extract_speech_asr(str(audio_path), model_name="medium")
    Path(CAPTIONS_DIR / f"scene_{idx:02d}_asr.txt").write_text(scene["speech"], encoding="utf-8")

    # AST (non-speech audio)
    scene["natural_audio_labels"] = extract_natural_audio_labels(str(audio_path), clip_sec=2, device="cpu", threshold=0.3)
    Path(CAPTIONS_DIR / f"scene_{idx:02d}_audio_labels.json").write_text(
        json.dumps(scene["natural_audio_labels"], indent=2), encoding="utf-8"
    )

    # Compose final caption (BLIP + ASR + AST) with "+" separator
    blip_caption = scene.get("blip_caption", "").strip()
    speech_caption = scene.get("speech", "").strip()
    audio_labels_caption = ', '.join([lbl for clip in scene.get("natural_audio_labels", []) for lbl in clip["labels"]])
    scene["final_caption"] = " + ".join(filter(None, [
        f"BLIP: {blip_caption}" if blip_caption else None,
        f"ASR: {speech_caption}" if speech_caption else None,
        f"AST: {audio_labels_caption}" if audio_labels_caption else None
    ]))
    Path(CAPTIONS_DIR / f"scene_{idx:02d}_full_caption.txt").write_text(scene["final_caption"], encoding="utf-8")

    # Profiling
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / 1024**2
    cpu_percent = psutil.cpu_percent(interval=0.5)
    gpu_util = torch.cuda.memory_allocated(0)/torch.cuda.max_memory_allocated(0)*100 if torch.cuda.is_available() else 0
    vram_usage = torch.cuda.memory_allocated(0)/1024**2 if torch.cuda.is_available() else 0
    scene_metrics.append({
        "scene_index": idx,
        "scene_duration_sec": end_sec - start_sec,
        "ram_usage_mb": ram_usage,
        "cpu_percent": cpu_percent,
        "gpu_util_percent": gpu_util,
        "vram_usage_mb": vram_usage
    })

# ----------------------------
# SAVE METRICS
# ----------------------------
total_time = time.time() - video_start_time
Path(CAPTIONS_DIR / "audio_processing_metrics.json").write_text(
    json.dumps({"total_time_sec": total_time, "scenes": scene_metrics}, indent=2), encoding="utf-8"
)
print(f"Total video processing time: {total_time:.2f} sec")

# ----------------------------
# COMBINE ALL SCENE CAPTIONS
# ----------------------------
final_json_path = CAPTIONS_DIR / "final_scenes.json"
if not final_json_path.exists():
    combined_output = {"video": str(VIDEO_PATH), "total_scenes": len(captioned_scenes), "scenes": []}
    for scene in captioned_scenes:
        idx = scene["scene_index"]
        final_cap_file = CAPTIONS_DIR / f"scene_{idx:02d}_full_caption.txt"
        caption_text = final_cap_file.read_text(encoding="utf-8").strip() if final_cap_file.exists() else ""
        combined_output["scenes"].append({
            "scene_index": idx,
            "start_sec": float(scene["start_seconds"]),
            "end_sec": float(scene["end_seconds"]),
            "caption": caption_text
        })
    final_json_path.write_text(json.dumps(combined_output, indent=2), encoding="utf-8")

    # Save TXT version
    final_txt_path = CAPTIONS_DIR / "final_scenes.txt"
    with final_txt_path.open("w", encoding="utf-8") as f:
        for s in combined_output["scenes"]:
            f.write(f"Scene {s['scene_index']} ({s['start_sec']:.2f}-{s['end_sec']:.2f} sec):\n")
            f.write(s["caption"] + "\n\n")
    print("Final combined captions saved")
else:
    print("Final scene JSON already exists — skipping recombination")
