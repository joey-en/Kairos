# metrics.py
import time
import psutil
import torch

try:
    import GPUtil
except ImportError:
    GPUtil = None

def get_cpu_usage():
    """Return instant CPU usage percentage."""
    return psutil.cpu_percent(interval=None)

def get_ram_usage():
    """Return used RAM in GB."""
    mem = psutil.virtual_memory()
    return mem.used / (1024 ** 3)   # GB

def get_gpu_metrics():
    """Return GPU utilization and VRAM used in GB. Safe fallback for CPU-only systems."""
    try:
        if not torch.cuda.is_available() or GPUtil is None:
            return {"gpu_util": 0.0, "vram_used": 0.0}
        gpu = GPUtil.getGPUs()[0]
        return {
            "gpu_util": gpu.load * 100,             # %
            "vram_used": gpu.memoryUsed / 1024      # GB
        }
    except Exception:
        return {"gpu_util": 0.0, "vram_used": 0.0}

class MetricsLogger:
    """Logger for CPU, RAM, GPU metrics over pipeline steps."""

    def __init__(self):
        self.start_time = time.time()
        self.steps = []

    def log_step(self, name: str):
        """Log metrics at a specific step."""
        cpu = get_cpu_usage()
        ram = get_ram_usage()
        gpu = get_gpu_metrics()

        self.steps.append({
            "step": name,
            "cpu": cpu,
            "ram": ram,
            "gpu_util": gpu["gpu_util"],
            "vram": gpu["vram_used"],
            "timestamp": time.time() - self.start_time
        })

    def save(self, filepath="metrics.txt"):
        """Save all logged metrics to a UTF-8 encoded text file."""
        total_time = time.time() - self.start_time

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"TOTAL PIPELINE TIME: {total_time:.2f} sec\n\n")

            for s in self.steps:
                f.write(
                    f"[{s['timestamp']:.2f}s] {s['step']} → "
                    f"CPU {s['cpu']}%, RAM {s['ram']:.2f}GB, "
                    f"GPU {s['gpu_util']:.1f}%, VRAM {s['vram']:.2f}GB\n"
                )
