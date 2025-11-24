import time
import functools
import os, sys, platform, subprocess
import json
import psutil
import torch
import pynvml

def safe_nvml_init():
    try:
        pynvml.nvmlInit()
        return True
    except:
        return False

NVML_AVAILABLE = safe_nvml_init()


def get_system_context():
    """
    Returns a dictionary summarizing the current computer's hardware,
    OS details, Python environment, and GPU (if available).
    """

    # --- Basic OS & Machine Info ---
    uname = platform.uname()
    system = {
        "os": f"{uname.system} {uname.release}",
        "os_version": uname.version,
        "machine_type": uname.machine,
        "hostname": uname.node,
        "python_version": sys.version.split()[0],
    }

    # --- CPU Info ---
    cpu_info = {
        "cpu_model": uname.processor or platform.processor(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_frequency_MHz": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
    }

    # --- RAM Info ---
    svmem = psutil.virtual_memory()
    ram_info = {
        "total_RAM_GB": round(svmem.total / (1024**3), 2),
        "available_RAM_GB": round(svmem.available / (1024**3), 2),
        "used_RAM_GB": round(svmem.used / (1024**3), 2),
        "RAM_usage_percent": svmem.percent,
    }

    # --- Disk Info ---
    disk = psutil.disk_usage("/")
    disk_info = {
        "disk_total_GB": round(disk.total / (1024**3), 2),
        "disk_used_GB": round(disk.used / (1024**3), 2),
        "disk_free_GB": round(disk.free / (1024**3), 2),
        "disk_usage_percent": disk.percent,
    }

    # --- GPU Info (NVIDIA only, via nvidia-smi) ---
    try:
        gpu_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        gpu_name, mem_total, mem_used, driver = gpu_output.strip().split(", ")
        gpu_info = {
            "gpu_model": gpu_name,
            "gpu_memory_total_MB": int(mem_total),
            "gpu_memory_used_MB": int(mem_used),
            "gpu_driver_version": driver,
        }
    except Exception:
        gpu_info = {"gpu_model": None}

    # --- Combine ---
    return {
        "os_info": system,
        "cpu_info": cpu_info,
        "ram_info": ram_info,
        "disk_info": disk_info,
        "gpu_info": gpu_info,
    }

def initiate_log(video_path, run_description):
    return {
        "run_description": run_description,
        "video_path": video_path,
        "start_process": time.time(),
        "computer": get_system_context(),
    }

def get_gpu_stats():
    """
    Return GPU stats if possible.
    NVML -> full stats
    PyTorch -> partial stats
    Otherwise -> returns []
    """

    # -------------------------
    # NVML path: full info
    # -------------------------
    if NVML_AVAILABLE:
        gpus = []
        try:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()

                gpus.append({
                    "id": i,
                    "name": name,
                    "memory_used_MB": mem.used // (1024**2),
                    "memory_total_MB": mem.total // (1024**2),
                    "gpu_util_percent": util.gpu,
                    "mem_util_percent": util.memory
                })
            return gpus
        except:
            pass  # continue to fallback

    # -------------------------
    # PyTorch CUDA fallback
    # -------------------------
    if torch.cuda.is_available():
        try:
            return [{
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_used_MB": torch.cuda.memory_allocated(i) // (1024**2),
                "memory_total_MB": torch.cuda.get_device_properties(i).total_memory // (1024**2),
                "gpu_util_percent": None,      # not available
                "mem_util_percent": None,      # not available
            } for i in range(torch.cuda.device_count())]
        except:
            pass

    # -------------------------
    # No GPU available
    # -------------------------
    return []


def detect_device_used():
    """
    Detect which device (CPU or GPU) is being used.
    Returns a dictionary with device information.
    """
    device_info = {
        "device_type": "cpu",
        "device_name": "CPU",
        "device_index": None,
    }
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Check if any CUDA memory is allocated (indicates GPU usage)
        total_allocated = sum(
            torch.cuda.memory_allocated(i) 
            for i in range(torch.cuda.device_count())
        )
        
        if total_allocated > 0:
            # GPU is being used - find which one has the most memory allocated
            max_mem = 0
            device_idx = 0
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.memory_allocated(i)
                if mem > max_mem:
                    max_mem = mem
                    device_idx = i
            
            device_info = {
                "device_type": "cuda",
                "device_name": torch.cuda.get_device_name(device_idx),
                "device_index": device_idx,
            }
        else:
            # CUDA available but no memory allocated yet
            # Check if we can determine from PyTorch's current device
            # This handles cases where models are loaded but haven't run inference yet
            try:
                current_device = torch.cuda.current_device()
                device_info = {
                    "device_type": "cuda",
                    "device_name": torch.cuda.get_device_name(current_device),
                    "device_index": current_device,
                }
            except:
                # Fallback to CPU - CUDA available but no device context set
                device_info = {
                    "device_type": "cpu",
                    "device_name": "CPU",
                    "device_index": None,
                }
    else:
        # CUDA not available - definitely using CPU
        device_info = {
            "device_type": "cpu",
            "device_name": "CPU",
            "device_index": None,
        }
    
    return device_info

def log_step():
    """Decorator that logs CPU, RAM, GPU, IO, runtime and returns (output, log_dict)."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            process = psutil.Process(os.getpid())

            # --- CPU / RAM / GPU / IO before ---
            cpu_before = time.process_time()
            ram_before = process.memory_info().rss // (1024 ** 2)
            io_before = process.io_counters()
            gpu_before = get_gpu_stats()
            device_before = detect_device_used()

            # CUDA memory before
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                cuda_before = [
                    torch.cuda.memory_allocated(i) // (1024 ** 2)
                    for i in range(torch.cuda.device_count())
                ]
            else:
                cuda_before = None

            # Print device info before function call
            func_name = func.__name__.replace("_log", "").replace("_", " ").title()
            device_str = f"{device_before['device_type'].upper()}"
            if device_before['device_type'] == 'cuda':
                device_str += f" ({device_before['device_name']}, GPU {device_before['device_index']})"
            print(f"\n[{func_name}] Starting... Device: {device_str}")

            # --- Run function ---
            t0 = time.time()
            output = func(*args, **kwargs)
            t1 = time.time()

            # --- CPU / RAM / GPU / IO after ---
            cpu_after = time.process_time()
            ram_after = process.memory_info().rss // (1024 ** 2)
            io_after = process.io_counters()
            gpu_after = get_gpu_stats()
            device_after = detect_device_used()

            # CUDA memory after + peak
            if torch.cuda.is_available():
                cuda_after = [
                    torch.cuda.memory_allocated(i) // (1024 ** 2)
                    for i in range(torch.cuda.device_count())
                ]
                cuda_peak = [
                    torch.cuda.max_memory_allocated(i) // (1024 ** 2)
                    for i in range(torch.cuda.device_count())
                ]
            else:
                cuda_after = cuda_peak = None

            # Determine primary device used during this step
            # If CUDA memory increased, GPU was likely used
            # Otherwise, prefer device_after (current state) but fallback to device_before if needed
            device_used = device_after
            if torch.cuda.is_available() and cuda_before is not None and cuda_after is not None:
                # Check if CUDA memory increased significantly (indicates GPU usage)
                total_before = sum(cuda_before) if isinstance(cuda_before, list) else 0
                total_after = sum(cuda_after) if isinstance(cuda_after, list) else 0
                if total_after > total_before + 10:  # At least 10MB increase indicates GPU usage
                    device_used = device_after
                elif device_after["device_type"] == "cpu" and device_before["device_type"] == "cpu":
                    # Both before and after are CPU - definitely CPU
                    device_used = device_after
                elif device_after["device_type"] == "cpu":
                    # After is CPU but before was GPU - might be CPU-only step
                    # Use CPU to indicate this step didn't use GPU
                    device_used = device_after
            elif device_after["device_type"] == "cpu":
                # No CUDA available or no memory tracking - use CPU
                device_used = device_after

            # Print device info after function call
            device_str = f"{device_used['device_type'].upper()}"
            if device_used['device_type'] == 'cuda':
                device_str += f" ({device_used['device_name']}, GPU {device_used['device_index']})"
            elapsed = round(t1 - t0, 2)
            print(f"[{func_name}] Completed in {elapsed}s. Device used: {device_str}")

            # Build log entry
            log_entry = {
                "wall_time_sec": round(t1 - t0, 5),
                "cpu_time_sec": round(cpu_after - cpu_before, 5),
                "ram_before_MB": ram_before,
                "ram_after_MB": ram_after,
                "ram_used_MB": ram_after - ram_before,
                "io_read_MB": (io_after.read_bytes - io_before.read_bytes) / (1024**2),
                "io_write_MB": (io_after.write_bytes - io_before.write_bytes) / (1024**2),
                "gpu_before": gpu_before,
                "gpu_after": gpu_after,
                "cuda_before_MB": cuda_before,
                "cuda_after_MB": cuda_after,
                "cuda_peak_MB": cuda_peak,
                "device_used": device_used,
            }

            return output, log_entry

        return wrapper
    return decorator

def complete_log(log, steps, vid_len, scene_num, vid_df=None):
    new_log =  {
        "run_description": log["run_description"],
        "video_path": log["video_path"],
        "video_length": vid_len,
        "total_process_sec": time.time() - log["start_process"],
        "scene_number": scene_num,
        "start_process": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(log["start_process"])),
        "end_process": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
        "computer": get_system_context(),
        "steps": steps
    }
    if vid_df is not None:
        new_log["scenes"] = vid_df
    return new_log

def save_log(data, folder="logs", filename="log"):
    """
    Saves any serializable data to a JSON file.
    - Creates folder if it does not exist
    - Automatically appends timestamp to filename
    - Returns the full path to the saved file
    """
    
    os.makedirs(folder, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(folder, f"{filename}_{timestamp}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"Log saved to: {file_path}")
    return file_path


# ================================================================================================
# WRAPPER FUNCTIONS
# ================================================================================================
from src.scene_cutting import get_scene_list
from src.frame_sampling import sample_frames
from src.frame_captioning_blip import caption_frames
from src.frame_obj_d_yolo import detect_object_yolo
from src.scene_description import describe_scenes
from src.audio_natural import extract_sounds
from src.audio_speech import extract_speech
from src.debug_utils import save_clips, save_vid_df


@log_step()
def get_scene_list_log(*args, **kwargs):
    return get_scene_list(*args, **kwargs)

@log_step()
def save_clips_log(*args, **kwargs):
    return save_clips(*args, **kwargs)

@log_step()
def sample_frames_log(*args, **kwargs):
    return sample_frames(*args, **kwargs)

@log_step()
def caption_frames_log(*args, **kwargs):
    return caption_frames(*args, **kwargs)
    
@log_step()
def detect_object_yolo_log(*args, **kwargs):
    return detect_object_yolo(*args, **kwargs)

@log_step()
def describe_scenes_log(*args, **kwargs):
    return describe_scenes(*args, **kwargs)

@log_step()
def extract_sounds_log(*args, **kwargs):
    return extract_sounds(*args, **kwargs)

@log_step()
def extract_speech_log(*args, **kwargs):
    return extract_speech(*args, **kwargs)

@log_step()
def save_vid_df_log(*args, **kwargs):
    return save_vid_df(*args, **kwargs)

# ================================================================================================
