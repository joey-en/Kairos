import time
import psutil
import torch  # for GPU and VRAM
import os
from pathlib import Path
import json
import tempfile
import subprocess

from src.scene_cutting import get_scene_list
from src.frame_sampling import sample_frames
from src.frame_captioning_blip import caption_frames
# from src.frame_captioning_heavy import refine_caption_frames   
from src.debug_utils import see_first_scene, see_scenes_cuts
from src.audio_asr import extract_speech_asr
from src.audio_natural import extract_natural_audio_labels_ast

# --- Setup output dirs ---
os.makedirs("./output", exist_ok=True)
os.makedirs("./output/frames", exist_ok=True)
os.makedirs("./output/audio", exist_ok=True)
os.makedirs("./output/captions", exist_ok=True)

test_video = r'Videos\SpongeBob SquarePants - Writing Essay - Some of These - Meme Source.mp4'

scenes = get_scene_list(test_video)
see_scenes_cuts(scenes)

scenes_with_frames = sample_frames(
    input_video_path=test_video,
    scenes=scenes,
    num_frames=2,
    output_dir="./output/frames",
)
# Has these errors (we need to check why):
# [h264 @ 000002602f6b6600] mmco: unref short failure
# [h264 @ 000002602fc03780] mmco: unref short failure

captioned_scenes = caption_frames(
    scenes=scenes_with_frames,
    max_length=30,
    num_beams=4,
    do_sample=False,
    debug=True,
    prompt="a video frame of"
)

# refined_scenes = refine_caption_frames(
#     scenes=captioned_scenes,
#     num_prev=1,
#     num_next=1,
#     extra_instruction=(
#         "Using the image and these captions as temporal context, "
#         "write ONE concise sentence describing what is happening "
#         "in this frame, focusing on new details or clarifications."
#     ),
#     do_sample=False,
#     debug=True,
# )

# see_first_scene(refined_scenes)

# --- Organize frame data safely ---
for scene in captioned_scenes:
    frames = scene.get("frames", [])
    frame_captions = scene.get("frame_captions", [])
    frame_paths = scene.get("frame_paths", [])
    scene["frames"] = [{"image": f, "path": frame_paths[idx] if idx < len(frame_paths) else "", "caption": frame_captions[idx] if idx < len(frame_captions) else ""} for idx, f in enumerate(frames)]
    scene["blip_caption"] = " ".join([f["caption"] for f in scene["frames"]])

hf_token = os.getenv("HF_TOKEN")

# --- Profiling variables ---
video_start_time = time.time()
scene_metrics = []

# --- Process each scene: audio, ASR, AST ---
for scene in captioned_scenes:
    scene_start_time = time.time()

    start_sec, end_sec = scene["start_seconds"], scene["end_seconds"]
    scene_audio_path = Path(f"./output/audio/scene_{scene['scene_index']}.wav")
    cmd = f'ffmpeg -y -i "{test_video}" -ss {start_sec} -to {end_sec} -ar 16000 -ac 1 -vn "{scene_audio_path}"'
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    scene["audio_path"] = str(scene_audio_path)

    # ASR
    scene["speech"] = extract_speech_asr(str(scene_audio_path), model_name="medium")
    Path(f"./output/captions/scene_{scene['scene_index']}_asr.txt").write_text(scene["speech"], encoding="utf-8")

    # AST
    scene["natural_audio_labels"] = extract_natural_audio_labels_ast(str(scene_audio_path), clip_sec=2, device="cpu", hf_token=hf_token)
    Path(f"./output/captions/scene_{scene['scene_index']}_audio_labels.json").write_text(json.dumps(scene["natural_audio_labels"], indent=2), encoding="utf-8")

    # Compose final caption
    blip_caption = scene.get("blip_caption", "")
    audio_label_names = [label for clip in scene["natural_audio_labels"] for label in clip["labels"]]
    scene["final_caption"] = f"{blip_caption} + {scene['speech']} + {', '.join(audio_label_names)}"
    Path(f"./output/captions/scene_{scene['scene_index']}_full_caption.txt").write_text(scene["final_caption"], encoding="utf-8")

    # --- Profiling per scene ---
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / 1024**2  # MB
    cpu_percent = psutil.cpu_percent(interval=0.5)  # over 0.5 sec
    gpu_util = torch.cuda.is_available() and torch.cuda.memory_allocated(0) / torch.cuda.max_memory_allocated(0) * 100 if torch.cuda.is_available() else 0
    vram_usage = torch.cuda.memory_allocated(0)/1024**2 if torch.cuda.is_available() else 0  # MB

    scene_metrics.append({
        "scene_index": scene["scene_index"],
        "scene_duration_sec": scene["end_seconds"] - scene["start_seconds"],
        "processing_time_sec": time.time() - scene_start_time,
        "ram_usage_mb": ram_usage,
        "cpu_percent": cpu_percent,
        "gpu_util_percent": gpu_util,
        "vram_usage_mb": vram_usage
    })

# --- Total video processing time ---
total_time = time.time() - video_start_time

# --- Save profiling metrics ---
Path("./output/captions/audio_processing_metrics.json").write_text(
    json.dumps({"total_time_sec": total_time, "scenes": scene_metrics}, indent=2),
    encoding="utf-8"
)

print(f"Done! Total video processing time: {total_time:.2f} sec")


# =====================================================================
# 📌 FINAL POST-PROCESSING: Combine all scene captions into one file
# =====================================================================

final_json_path = Path("./output/captions/final_scenes.json")

# --- If final output exists, do NOT rerun heavy pipeline ---
if final_json_path.exists():
    print("Final scene JSON already exists — skipping recombination.")
else:
    combined_output = {
        "video": test_video,
        "total_scenes": len(captioned_scenes),
        "scenes": []
    }

    for scene in captioned_scenes:
        idx = scene["scene_index"]
        start_sec = scene["start_seconds"]
        end_sec = scene["end_seconds"]
        final_cap_file = Path(f"./output/captions/scene_{idx}_full_caption.txt")

        if final_cap_file.exists():
            caption_text = final_cap_file.read_text(encoding="utf-8").strip()
        else:
            caption_text = ""

        combined_output["scenes"].append({
            "scene_index": idx,
            "start_sec": float(start_sec),
            "end_sec": float(end_sec),
            "caption": caption_text
        })

    # --- Save to JSON ---
    final_json_path.write_text(
        json.dumps(combined_output, indent=2),
        encoding="utf-8"
    )

    # --- Save to TXT ---
    final_txt_path = Path("./output/captions/final_scenes.txt")
    with final_txt_path.open("w", encoding="utf-8") as f:
        for s in combined_output["scenes"]:
            f.write(f"Scene {s['scene_index']} ({s['start_sec']}–{s['end_sec']} sec):\n")
            f.write(s["caption"] + "\n\n")

    print("Final combined captions saved ✔️")
