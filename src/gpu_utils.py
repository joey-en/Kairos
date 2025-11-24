"""
GPU utility functions for selecting and managing GPU devices.
"""
import torch
import os
from typing import Optional, Union

def get_discrete_gpu_index() -> Optional[int]:
    """
    Attempts to identify the discrete GPU by looking for GPUs with more VRAM
    or by excluding integrated GPU names.
    
    Returns:
        int: GPU index of the discrete GPU, or None if not found
    """
    if not torch.cuda.is_available():
        return None
    
    # Common integrated GPU names (case-insensitive)
    integrated_keywords = [
        "intel", "iris", "uhd", "hd graphics", 
        "radeon graphics", "vega", "amd apu"
    ]
    
    discrete_keywords = [
        "nvidia", "geforce", "rtx", "gtx", "quadro", "tesla",
        "amd", "radeon rx", "radeon pro"
    ]
    
    gpu_count = torch.cuda.device_count()
    
    if gpu_count == 0:
        return None
    elif gpu_count == 1:
        return 0  # Only one GPU available
    
    # Multiple GPUs - try to identify discrete one
    discrete_candidates = []
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i).lower()
        total_memory = torch.cuda.get_device_properties(i).total_memory
        
        # Check if it's likely discrete
        is_integrated = any(keyword in gpu_name for keyword in integrated_keywords)
        is_discrete = any(keyword in gpu_name for keyword in discrete_keywords)
        
        if is_discrete and not is_integrated:
            discrete_candidates.append((i, total_memory))
        elif not is_integrated and total_memory > 2 * 1024**3:  # > 2GB VRAM
            # Likely discrete if it has significant VRAM
            discrete_candidates.append((i, total_memory))
    
    if discrete_candidates:
        # Return the GPU with the most VRAM
        discrete_candidates.sort(key=lambda x: x[1], reverse=True)
        return discrete_candidates[0][0]
    
    # Fallback: return GPU with most VRAM
    gpus_with_memory = [
        (i, torch.cuda.get_device_properties(i).total_memory)
        for i in range(gpu_count)
    ]
    gpus_with_memory.sort(key=lambda x: x[1], reverse=True)
    return gpus_with_memory[0][0]


def get_device(gpu_index: Optional[int] = None, prefer_discrete: bool = True) -> Union[str, torch.device]:
    """
    Get the appropriate device for PyTorch operations.
    
    Args:
        gpu_index: Specific GPU index to use (0, 1, etc.). If None, auto-selects.
        prefer_discrete: If True and gpu_index is None, tries to select discrete GPU.
    
    Returns:
        torch.device: The device to use (e.g., "cuda:0", "cuda:1", or "cpu")
    
    Note:
        If CUDA is not available, this will return CPU. Check PyTorch installation
        if you have an NVIDIA GPU but CUDA is not detected (likely CPU-only PyTorch).
    """
    if not torch.cuda.is_available():
        # Check if PyTorch was built with CUDA but GPU not detected
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            import warnings
            warnings.warn(
                f"PyTorch was built with CUDA {torch.version.cuda} but no GPU detected. "
                "Check GPU drivers or use CPU. See INSTALL_CUDA_PYTORCH.md for help.",
                UserWarning
            )
        return torch.device("cpu")
    
    # Use environment variable if set (highest priority)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        # When CUDA_VISIBLE_DEVICES is set, PyTorch remaps indices
        # So we use cuda:0 which refers to the first visible device
        return torch.device("cuda:0")
    
    # Use specified GPU index
    if gpu_index is not None:
        if gpu_index >= torch.cuda.device_count():
            raise ValueError(f"GPU index {gpu_index} is out of range. Available GPUs: {torch.cuda.device_count()}")
        return torch.device(f"cuda:{gpu_index}")
    
    # Auto-select discrete GPU if requested
    if prefer_discrete:
        discrete_idx = get_discrete_gpu_index()
        if discrete_idx is not None:
            return torch.device(f"cuda:{discrete_idx}")
    
    # Default to first GPU
    return torch.device("cuda:0")


def print_gpu_info():
    """Print information about available GPUs."""
    if not torch.cuda.is_available():
        print("No CUDA GPUs available.")
        # Check if PyTorch has CUDA support but GPU not detected
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            print(f"\nWARNING: PyTorch was built with CUDA {torch.version.cuda} but GPU not detected!")
            print("Possible issues:")
            print("  1. GPU drivers not installed")
            print("  2. CUDA version mismatch")
            print("  3. GPU not properly connected")
            print("\nRun 'nvidia-smi' to check GPU status.")
        else:
            print("\nPyTorch was installed WITHOUT CUDA support (CPU-only version).")
            print("To use your RTX 5090, install PyTorch with CUDA:")
            print("  See INSTALL_CUDA_PYTORCH.md for instructions")
        return
    
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print()
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print()
    
    discrete_idx = get_discrete_gpu_index()
    if discrete_idx is not None:
        print(f"Recommended discrete GPU: GPU {discrete_idx} ({torch.cuda.get_device_name(discrete_idx)})")
    else:
        print("Could not automatically detect discrete GPU.")

