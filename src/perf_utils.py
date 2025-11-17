import time
import psutil
import os

def measure_performance(func):
    """
    Decorator to measure execution time and memory usage of a function.
    """
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024**2)  # MB
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        mem_after = process.memory_info().rss / (1024**2)  # MB

        print(f"[PERF] {func.__name__} -> Time: {end_time - start_time:.2f}s, "
              f"Memory: {mem_after - mem_before:.2f} MB (delta)")

        return result
    return wrapper

def get_gpu_stats():
    try:
        import torch
        if torch.cuda.is_available():
            util = torch.cuda.utilization() if hasattr(torch.cuda, "utilization") else 0
            mem = torch.cuda.memory_allocated() / 1024 / 1024
            return util, mem
    except Exception:
        pass
    return 0, 0