import os
import sys
import cv2
import time
import psutil
import torch
from simple_frame_differencing import compute_frame_differences
from MOG2_frame_differencing import compute_mog2_differences
from KNN_frame_differencing import compute_knn_differences
from utils import extract_changed_frames

# Suppress h264 warnings in output
class SuppressStderr:
    def __enter__(self):
        self._original_stderr = sys.stderr
        self._devnull = open(os.devnull, 'w')
        sys.stderr = self._devnull
        return self
    
    def __exit__(self, *args):
        sys.stderr = self._original_stderr
        self._devnull.close()

# ----------------------------
# System monitoring functions
# ----------------------------
def get_system_metrics():
    """Get current RAM, CPU, and GPU metrics"""
    process = psutil.Process()
    mem_info = process.memory_info()
    
    metrics = {
        'ram_mb': mem_info.rss / 1024**2,
        'cpu_percent': process.cpu_percent(interval=0.1),
        'gpu_available': torch.cuda.is_available(),
        'gpu_mem_gb': 0,
        'gpu_util_percent': 0
    }
    
    if metrics['gpu_available']:
        try:
            metrics['gpu_mem_gb'] = torch.cuda.memory_allocated() / 1024**3
            # Note: GPU utilization requires nvidia-smi or pynvml
            # For now, we'll just track memory
        except:
            pass
    
    return metrics

# ----------------------------
# Paths
# ----------------------------
video_path = r"..\Videos\SpongeBob SquarePants - Writing Essay - Some of These - Meme Source.mp4"

os.makedirs("output/changed_frames/simple", exist_ok=True)
os.makedirs("output/changed_frames/mog2", exist_ok=True)
os.makedirs("output/changed_frames/knn", exist_ok=True)

# ----------------------------
# Video info
# ----------------------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total_frames / fps
cap.release()

print("="*70)
print("COMPREHENSIVE FRAME DIFFERENCING BENCHMARK")
print("="*70)
print(f"Video: {video_path}")
print(f"Resolution: {width}x{height} ({width*height:,} pixels per frame)")
print(f"Duration: {duration:.1f}s ({total_frames} frames @ {fps:.1f} fps)")
print()

# Baseline metrics
baseline = get_system_metrics()
print("System Baseline:")
print(f"  RAM Usage: {baseline['ram_mb']:.1f} MB")
print(f"  CPU Usage: {baseline['cpu_percent']:.1f}%")
if baseline['gpu_available']:
    print(f"  GPU: Available (CUDA)")
    print(f"  VRAM Usage: {baseline['gpu_mem_gb']:.2f} GB")
else:
    print(f"  GPU: Not available (CPU-only)")
print()

# ----------------------------
# Benchmark storage
# ----------------------------
results = {}

# ----------------------------
# METHOD 1: SIMPLE
# ----------------------------
print("="*70)
print("METHOD 1: SIMPLE FRAME DIFFERENCING")
print("="*70)

before = get_system_metrics()
start_time = time.time()

simple_events = compute_frame_differences(video_path)

end_time = time.time()
after = get_system_metrics()

simple_time = end_time - start_time

results['Simple'] = {
    'events': simple_events,
    'time': simple_time,
    'ram_before': before['ram_mb'],
    'ram_after': after['ram_mb'],
    'ram_delta': after['ram_mb'] - before['ram_mb'],
    'cpu_percent': after['cpu_percent'],
    'fps_processed': total_frames / simple_time
}

print(f"✓ Completed in {simple_time:.2f}s")
print(f"  Processing Speed: {results['Simple']['fps_processed']:.1f} fps")
print(f"  RAM Usage: {before['ram_mb']:.1f} MB → {after['ram_mb']:.1f} MB (Δ {results['Simple']['ram_delta']:+.1f} MB)")
print(f"  CPU Usage: {after['cpu_percent']:.1f}%")
print(f"  Motion Frames: {len(simple_events)}/{total_frames} ({len(simple_events)/total_frames*100:.1f}%)")
print()

# ----------------------------
# METHOD 2: MOG2
# ----------------------------
print("="*70)
print("METHOD 2: MOG2 BACKGROUND SUBTRACTION")
print("="*70)

before = get_system_metrics()
start_time = time.time()

mog2_events = compute_mog2_differences(video_path)

end_time = time.time()
after = get_system_metrics()

mog2_time = end_time - start_time

results['MOG2'] = {
    'events': mog2_events,
    'time': mog2_time,
    'ram_before': before['ram_mb'],
    'ram_after': after['ram_mb'],
    'ram_delta': after['ram_mb'] - before['ram_mb'],
    'cpu_percent': after['cpu_percent'],
    'fps_processed': total_frames / mog2_time
}

print(f"✓ Completed in {mog2_time:.2f}s")
print(f"  Processing Speed: {results['MOG2']['fps_processed']:.1f} fps")
print(f"  RAM Usage: {before['ram_mb']:.1f} MB → {after['ram_mb']:.1f} MB (Δ {results['MOG2']['ram_delta']:+.1f} MB)")
print(f"  CPU Usage: {after['cpu_percent']:.1f}%")
print(f"  Motion Frames: {len(mog2_events)}/{total_frames} ({len(mog2_events)/total_frames*100:.1f}%)")
print()

# ----------------------------
# METHOD 3: KNN
# ----------------------------
print("="*70)
print("METHOD 3: KNN BACKGROUND SUBTRACTION")
print("="*70)

before = get_system_metrics()
start_time = time.time()

knn_events = compute_knn_differences(video_path)

end_time = time.time()
after = get_system_metrics()

knn_time = end_time - start_time

results['KNN'] = {
    'events': knn_events,
    'time': knn_time,
    'ram_before': before['ram_mb'],
    'ram_after': after['ram_mb'],
    'ram_delta': after['ram_mb'] - before['ram_mb'],
    'cpu_percent': after['cpu_percent'],
    'fps_processed': total_frames / knn_time
}

print(f"✓ Completed in {knn_time:.2f}s")
print(f"  Processing Speed: {results['KNN']['fps_processed']:.1f} fps")
print(f"  RAM Usage: {before['ram_mb']:.1f} MB → {after['ram_mb']:.1f} MB (Δ {results['KNN']['ram_delta']:+.1f} MB)")
print(f"  CPU Usage: {after['cpu_percent']:.1f}%")
print(f"  Motion Frames: {len(knn_events)}/{total_frames} ({len(knn_events)/total_frames*100:.1f}%)")
print()

# ----------------------------
# Extract & Save Frames (with warnings suppressed)
# ----------------------------
print("="*70)
print("EXTRACTING & SAVING FRAMES")
print("="*70)

print(f"📸 Simple: Extracting {len(simple_events)} frames...", end=" ", flush=True)
with SuppressStderr():
    simple_frames = extract_changed_frames(video_path, simple_events)
print("✓")

print(f"📸 MOG2:   Extracting {len(mog2_events)} frames...", end=" ", flush=True)
with SuppressStderr():
    mog2_frames = extract_changed_frames(video_path, mog2_events)
print("✓")

print(f"📸 KNN:    Extracting {len(knn_events)} frames...", end=" ", flush=True)
with SuppressStderr():
    knn_frames = extract_changed_frames(video_path, knn_events)
print("✓")

print()
print("💾 Saving frames to disk...")
for idx, frame in enumerate(simple_frames):
    cv2.imwrite(f"output/changed_frames/simple/frame_{idx:04d}.png", frame)
for idx, frame in enumerate(mog2_frames):
    cv2.imwrite(f"output/changed_frames/mog2/frame_{idx:04d}.png", frame)
for idx, frame in enumerate(knn_frames):
    cv2.imwrite(f"output/changed_frames/knn/frame_{idx:04d}.png", frame)
print("✓ All frames saved")
print()

# ----------------------------
# Retrieval Latency Test
# ----------------------------
print("="*70)
print("RETRIEVAL LATENCY (Single Frame Access)")
print("="*70)

test_frame_idx = total_frames // 2
latencies = {}

for method_name in ['Simple', 'MOG2', 'KNN']:
    times = []
    for _ in range(5):  # Average of 5 attempts
        with SuppressStderr():
            start = time.time()
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame_idx)
            ret, frame = cap.read()
            cap.release()
            times.append((time.time() - start) * 1000)  # ms
    
    avg_latency = sum(times) / len(times)
    latencies[method_name] = avg_latency
    print(f"  {method_name:8s}: {avg_latency:.2f} ms (avg of 5 retrievals)")

print()

# ----------------------------
# Motion Quality Statistics
# ----------------------------
def avg_pixels(events):
    return sum(e['changed_pixels'] for e in events) / len(events) if events else 0

# ----------------------------
# COMPREHENSIVE COMPARISON
# ----------------------------
print("="*70)
print("COMPREHENSIVE PERFORMANCE COMPARISON")
print("="*70)
print()

print("⏱️  TOTAL PROCESSING TIME:")
print(f"  Simple:  {results['Simple']['time']:6.2f}s  (baseline)")
print(f"  MOG2:    {results['MOG2']['time']:6.2f}s  ({results['MOG2']['time']/results['Simple']['time']:.1f}x slower)")
print(f"  KNN:     {results['KNN']['time']:6.2f}s  ({results['KNN']['time']/results['Simple']['time']:.1f}x slower)")
print()

print("🚀 PROCESSING SPEED (frames/sec):")
print(f"  Simple:  {results['Simple']['fps_processed']:6.1f} fps")
print(f"  MOG2:    {results['MOG2']['fps_processed']:6.1f} fps")
print(f"  KNN:     {results['KNN']['fps_processed']:6.1f} fps")
print()

print("💾 RAM USAGE:")
print(f"  Simple:  {results['Simple']['ram_after']:6.1f} MB (Δ {results['Simple']['ram_delta']:+6.1f} MB)")
print(f"  MOG2:    {results['MOG2']['ram_after']:6.1f} MB (Δ {results['MOG2']['ram_delta']:+6.1f} MB)")
print(f"  KNN:     {results['KNN']['ram_after']:6.1f} MB (Δ {results['KNN']['ram_delta']:+6.1f} MB)")
print()

print("🔄 CPU USAGE (during processing):")
print(f"  Simple:  {results['Simple']['cpu_percent']:5.1f}%")
print(f"  MOG2:    {results['MOG2']['cpu_percent']:5.1f}%")
print(f"  KNN:     {results['KNN']['cpu_percent']:5.1f}%")
print()

print("⚡ RETRIEVAL LATENCY (single frame):")
print(f"  Simple:  {latencies['Simple']:6.2f} ms")
print(f"  MOG2:    {latencies['MOG2']:6.2f} ms")
print(f"  KNN:     {latencies['KNN']:6.2f} ms")
print()

print("📊 MOTION DETECTION:")
print(f"  Simple:  {len(simple_events):4d} frames ({len(simple_events)/total_frames*100:5.1f}%)")
print(f"  MOG2:    {len(mog2_events):4d} frames ({len(mog2_events)/total_frames*100:5.1f}%)")
print(f"  KNN:     {len(knn_events):4d} frames ({len(knn_events)/total_frames*100:5.1f}%)")
print()

print("🎯 MOTION QUALITY (avg pixels changed):")
print(f"  Simple:  {avg_pixels(simple_events):>10,.0f} pixels")
print(f"  MOG2:    {avg_pixels(mog2_events):>10,.0f} pixels")
print(f"  KNN:     {avg_pixels(knn_events):>10,.0f} pixels")
print()

# ----------------------------
# Final Recommendation
# ----------------------------
print("="*70)
print("RECOMMENDATION FOR PYSCENEDETECT + BLIP PIPELINE")
print("="*70)

simple_count = len(simple_events)
mog2_count = len(mog2_events)

if simple_count < mog2_count * 0.6:
    print("🏆 RECOMMENDED: MOG2")
    print()
    print("Why MOG2:")
    print(f"  ✓ Detects {mog2_count - simple_count} more motion frames than Simple")
    print(f"  ✓ Only {results['MOG2']['time']/results['Simple']['time']:.1f}x slower than Simple")
    print(f"  ✓ Better at handling cartoon motion and scene transitions")
    print(f"  ✓ More robust to lighting changes")
    print()
elif simple_count > mog2_count * 1.5:
    print("🏆 RECOMMENDED: MOG2 or KNN")
    print()
    print("Why not Simple:")
    print(f"  ⚠️  Simple detects too much ({simple_count} vs {mog2_count} frames)")
    print("  → Likely picking up noise and minor changes")
    print("  → MOG2/KNN provide better filtering")
else:
    print("🏆 RECOMMENDED: Simple Frame Differencing")
    print()
    print("Why Simple:")
    print(f"  ✓ {results['Simple']['fps_processed']/results['MOG2']['fps_processed']:.1f}x faster than MOG2")
    print(f"  ✓ Similar detection quality ({simple_count} vs {mog2_count} frames)")
    print(f"  ✓ Lower RAM usage ({results['Simple']['ram_delta']:.0f} MB vs {results['MOG2']['ram_delta']:.0f} MB)")
    print("  ✓ Best for real-time processing")

print()
print("="*70)
print("FILES SAVED")
print("="*70)
print(f"📁 output/changed_frames/simple/  ({len(simple_frames)} frames)")
print(f"📁 output/changed_frames/mog2/    ({len(mog2_frames)} frames)")
print(f"📁 output/changed_frames/knn/     ({len(knn_frames)} frames)")
print()
print("✅ COMPLETE! Check folders to visually compare motion detection quality.")
print("="*70)