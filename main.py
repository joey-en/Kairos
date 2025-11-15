import os
import time
import json
import psutil
import torch
from src.scene_cutting import get_scene_list
from src.frame_sampling import sample_frames
from src.captioning.frame_captioning_blip import BLIPCaptioner
from src.detection.yolo_detector import YOLODetector
from src.captioning.llm_fusion_caption import LLMSegmentCaptioner

# ------------------------
# Helpers
# ------------------------
process = psutil.Process(os.getpid())
def mem_mb():
    return process.memory_info().rss / 1024 / 1024

# ------------------------
# Configuration
# ------------------------
video_path = "Videos/Watch Malala Yousafzai's Nobel Peace Prize acceptance speech.mp4" 
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Load models
# ------------------------
print("[INFO] Loading BLIP model...")
blip_captioner = BLIPCaptioner(device=device)

print("[INFO] Loading YOLOv8 model...")
yolo = YOLODetector("yolov8s.pt")

print("[INFO] Loading Gemini LLM Segment Captioner...")
llm_captioner = LLMSegmentCaptioner(model="gemini-2.5-flash")

# ------------------------
# Pipeline start metrics
# ------------------------
pipeline_start = time.time()
pipeline_mem_start = mem_mb()

# ------------------------
# Scene detection
# ------------------------
t0_scene = time.time()
scenes = get_scene_list(video_path)
t1_scene = time.time() - t0_scene
print(f"[INFO] Detected {len(scenes)} scenes in {t1_scene:.2f}s")

# ------------------------
# Frame sampling
# ------------------------
t0_sample = time.time()
scenes_with_frames = sample_frames(video_path, scenes, num_frames=2)
t1_sample = time.time() - t0_sample
print(f"[INFO] Sampled frames in {t1_sample:.2f}s")

# ------------------------
# Process each scene
# ------------------------
results = []
overall_yolo_time = 0
overall_blip_time = 0
overall_segment_time = 0

for scene in scenes_with_frames:
    frames = scene["frames"]
    scene_index = scene.get("scene_index", "unknown")
    scene_results = {
        "scene_index": scene_index,
        "frames": [],
        "segment_caption": None,
        "metrics": {}
    }

    # Per-frame BLIP + YOLO
    frame_outputs = []
    for frame in frames:
        # YOLO detection
        t0_yolo = time.time()
        yolo_dets = yolo.detect(frame)  # returns dicts with 'label' and 'confidence'
        t1_yolo = time.time() - t0_yolo
        overall_yolo_time += t1_yolo

        # BLIP caption
        t0_blip = time.time()
        blip_cap = blip_captioner.caption(frame, prompt="a video frame of")
        t1_blip = time.time() - t0_blip
        overall_blip_time += t1_blip

        frame_outputs.append({
            "yolo_detections": yolo_dets,
            "blip_caption": blip_cap,
            "metrics": {
                "yolo_time": round(t1_yolo, 4),
                "blip_time": round(t1_blip, 4),
            }
        })

    scene_results["frames"] = frame_outputs

    # Segment-level LLM caption
    t0_segment = time.time()
    fusion_texts = []
    for fo in frame_outputs:
        dets = ", ".join([f"{d['label']} ({d['confidence']:.2f})" for d in fo["yolo_detections"]]) \
               if fo["yolo_detections"] else "no objects"
        fusion_texts.append(f"Objects: {dets}\nCaption: {fo['blip_caption']}")

    segment_caption = llm_captioner.describe_segment(fusion_texts)
    t1_segment = time.time() - t0_segment
    overall_segment_time += t1_segment

    scene_results["segment_caption"] = {
        "text": segment_caption,
        "time": round(t1_segment, 4)
    }

    results.append(scene_results)

# ------------------------
# Pipeline summary
# ------------------------
pipeline_time = time.time() - pipeline_start
pipeline_mem_end = mem_mb()

print("\n[PIPELINE SUMMARY]")
print(f"Scenes processed: {len(scenes)}")
print(f"Total frames: {sum(len(s['frames']) for s in results)}")
print(f"Pipeline runtime: {pipeline_time:.2f}s")
print(f"Memory usage delta: {pipeline_mem_end - pipeline_mem_start:.2f} MB")
print(f"Total YOLO time: {overall_yolo_time:.2f}s")
print(f"Total BLIP time: {overall_blip_time:.2f}s")
print(f"Total LLM segment caption time: {overall_segment_time:.2f}s")

# ------------------------
# Print all segment captions at the end
# ------------------------
print("\n[SEGMENT CAPTIONS]")
for scene in results:
    print(f"Scene {scene['scene_index']}:")
    print(scene["segment_caption"]["text"])
    print("---")

# ------------------------
# Save JSON
# ------------------------
with open("/output/llm_results/results_fusion.json", "w") as f:
    json.dump(results, f, indent=2)
print("[INFO] Results saved to results_fusion.json")
