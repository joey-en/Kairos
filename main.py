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
VIDEO = "Videos/Watch Malala Yousafzai's Nobel Peace Prize acceptance speech.mp4" 
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


pipeline_start = time.time()
pipeline_mem_start = mem_mb()

# Scene detection
s_t0 = time.time()
scenes = get_scene_list(VIDEO)
s_time = time.time() - s_t0
print(f"[INFO] Detected {len(scenes)} scenes in {s_time:.2f}s")

# Frame sampling
f_t0 = time.time()
scenes = sample_frames(VIDEO, scenes, num_frames=3, output_dir="output/frames")
f_time = time.time() - f_t0
print(f"[INFO] Sampled frames in {f_time:.2f}s")

results = []
YOLO_total = 0
BLIP_total = 0
SEG_total = 0

for sc in scenes:
    frames = sc["frames"]
    scene_idx = sc["scene_index"]
    scene_res = {"scene_index": scene_idx, "frames": [], "metrics": {}, "segment_caption": None}

    frame_fusions = []
    for fr in frames:
        dets, t_y, cpu_y, ram_y, gpu_y, vram_y = yolo.detect(fr)
        caption, t_b, cpu_b, ram_b, gpu_b, vram_b = blip.caption(fr, prompt="a video frame of")

        YOLO_total += t_y
        BLIP_total += t_b

        frame_fusions.append({
        "yolo_detections": dets,
        "blip_caption": caption,
        "metrics": {
        "yolo_time": t_y,
        "blip_time": t_b,
        "yolo_cpu": cpu_y,
        "yolo_ram": ram_y,
        "yolo_gpu": gpu_y,
        "yolo_vram": vram_y,
        "blip_cpu": cpu_b,
        "blip_ram": ram_b,
        "blip_gpu": gpu_b,
        "blip_vram": vram_b,
        }
    })

    fusion_texts = []
    for fo in frame_fusions:
        objs = ", ".join([f"{d['label']} ({d['confidence']:.2f})" for d in fo["yolo_detections"]]) or "no objects"
        fusion_texts.append(f"Objects: {objs}\nCaption: {fo['blip_caption']}")

    t0 = time.time()
    seg_cap = llm.describe_segment(fusion_texts)
    tseg = time.time() - t0
    SEG_total += tseg

    scene_res["frames"] = frame_fusions
    scene_res["segment_caption"] = {"text": seg_cap, "time": tseg}
    results.append(scene_res)

pipeline_time = time.time() - pipeline_start
pipeline_mem_end = mem_mb()


print("\n[SUMMARY]")
print(f"Scenes: {len(scenes)}")
print(f"Scene detection: {s_time:.2f}s")
print(f"Sampling: {f_time:.2f}s")
print(f"YOLO total: {YOLO_total:.2f}s")
print(f"BLIP total: {BLIP_total:.2f}s")
print(f"LLM total: {SEG_total:.2f}s")
print(f"Pipeline time: {pipeline_time:.2f}s")
print(f"Memory delta: {pipeline_mem_end - pipeline_mem_start:.2f} MB")

print("\n[SEGMENT CAPTIONS]")
for sc in results:
    print(f"Scene {sc['scene_index']}: {sc['segment_caption']['text']}\n---")

os.makedirs("output/llm_results", exist_ok=True)
with open("output/llm_results/results.json", "w") as f:
    json.dump(results, f, indent=2)
print("[INFO] Results saved to output/llm_results/results_fusion.json")
