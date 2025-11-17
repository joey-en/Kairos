# main.py
"""
Video-to-Text Pipeline (Scene-based)

Steps:
1. Scene detection
2. Frame sampling + BLIP captions
3. Audio extraction per scene
4. ASR (speech transcription) per scene
5. AST (natural/environmental audio labels) per scene
6. Save results:
   - scene_X.wav
   - scene_X_asr.txt
   - scene_X_audio_labels.json
   - scene_X_full_caption.txt
   - timings.txt (ASR + AST per scene)
"""

import os
import time
import json
from pathlib import Path

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
TIMINGS_FILE = OUTPUT_DIR / "timings.txt"

for d in [FRAMES_DIR, AUDIO_DIR, CAPTIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FFMPEG_PATH = get_ffmpeg_executable()

# Clear timings file if exists
TIMINGS_FILE.write_text("")

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
for scene in captioned_scenes:
    start_sec, end_sec = scene["start_seconds"], scene["end_seconds"]
    idx = scene["scene_index"]

    # --- Extract per-scene audio ---
    audio_path = AUDIO_DIR / f"scene_{idx:02d}.wav"
    extract_scene_audio_ffmpeg(str(VIDEO_PATH), str(audio_path), start_sec, end_sec)
    scene["audio_path"] = str(audio_path)

    # --- ASR ---
    speech_text, asr_timings = extract_speech_asr(str(audio_path), model_name="medium", enable_logs=False)
    scene["speech"] = speech_text
    scene["asr_timings"] = asr_timings

    Path(CAPTIONS_DIR / f"scene_{idx:02d}_asr.txt").write_text(scene["speech"], encoding="utf-8")

    # --- AST ---
    natural_labels, ast_timings = extract_natural_audio_labels(str(audio_path), clip_sec=2, device="cpu", threshold=0.3, enable_logs=False)
    scene["natural_audio_labels"] = natural_labels
    scene["ast_timings"] = ast_timings

    Path(CAPTIONS_DIR / f"scene_{idx:02d}_audio_labels.json").write_text(
        json.dumps(natural_labels, indent=2), encoding="utf-8"
    )

    # --- Final combined caption ---
    blip_caption = scene.get("blip_caption", "").strip()
    speech_caption = scene["speech"].strip()
    audio_labels_caption = ', '.join(lbl for clip in natural_labels for lbl in clip["labels"])

    scene["final_caption"] = " + ".join(
        list(filter(None, [
            f"BLIP: {blip_caption}" if blip_caption else None,
            f"ASR: {speech_caption}" if speech_caption else None,
            f"AST: {audio_labels_caption}" if audio_labels_caption else None,
        ]))
    )

    Path(CAPTIONS_DIR / f"scene_{idx:02d}_full_caption.txt").write_text(scene["final_caption"], encoding="utf-8")

    # --- Save timings per scene ---
    with open(TIMINGS_FILE, "a", encoding="utf-8") as f:
        f.write(f"Scene {idx} | {start_sec:.2f}-{end_sec:.2f} sec\n")
        f.write(f"  ASR duration: {asr_timings.get('asr_duration_sec', -1):.4f} sec\n")
        f.write(f"    Load audio: {asr_timings.get('load_audio_sec', -1):.4f} sec\n")
        f.write(f"    Noise reduction: {asr_timings.get('noise_reduction_sec', -1):.4f} sec\n")
        f.write(f"    VAD detection: {asr_timings.get('vad_detection_sec', -1):.4f} sec\n")
        f.write(f"    Chunk extraction: {asr_timings.get('extract_chunks_sec', -1):.4f} sec\n")
        f.write(f"    Whisper: {asr_timings.get('whisper_sec', -1):.4f} sec\n")
        f.write(f"  AST duration: {ast_timings.get('ast_duration_sec', -1):.4f} sec\n")
        f.write(f"    Load audio: {ast_timings.get('load_audio_sec', -1):.4f} sec\n")
        f.write(f"    VAD detection: {ast_timings.get('vad_detection_sec', -1):.4f} sec\n")
        f.write(f"    Mask speech: {ast_timings.get('mask_speech_sec', -1):.4f} sec\n")
        f.write(f"    Feature extraction: {ast_timings.get('feature_extract_sec', -1):.4f} sec\n")
        f.write(f"    AST forward: {ast_timings.get('ast_forward_sec', -1):.4f} sec\n")
        f.write("-"*60 + "\n")

print(f"\nAll scene processing complete. Timings saved to {TIMINGS_FILE}")