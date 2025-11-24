import av
import numpy as np
import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from src.gpu_utils import get_device
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Thread lock for AST GPU operations to prevent memory corruption
_ast_lock = threading.Lock()

# =========================================================
# Load Silero VAD
# =========================================================
silero_model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False
)
(get_speech_ts,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils


# =========================================================
# Load AST model
# =========================================================
AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_FE = AutoFeatureExtractor.from_pretrained(AST_MODEL_NAME)
AST_MODEL = AutoModelForAudioClassification.from_pretrained(AST_MODEL_NAME)
# Move AST model to GPU if available
device = get_device(prefer_discrete=True)
AST_MODEL = AST_MODEL.to(device)


# =========================================================
# Load audio directly from video using PyAV
# =========================================================
def load_audio_av(video_path, target_sr=16000):
    """
    Extract full audio from problematic MP4 files using PyAV + timestamp correction.
    """

    container = av.open(
        video_path,
        options={
            "fflags": "+genpts",
            "ignore_editlist": "1"
        }
    )

    audio_stream = next(s for s in container.streams if s.type == "audio")
    audio_stream.thread_type = "AUTO"

    samples = []

    for frame in container.decode(audio_stream):
        pcm = frame.to_ndarray()  # (channels, samples)
        pcm = pcm.mean(axis=0)    # stereo → mono
        samples.append(pcm)

    if not samples:
        return np.zeros(1, dtype=np.float32), target_sr

    audio = np.concatenate(samples).astype(np.float32)

    # Resample with librosa
    audio = librosa.resample(audio, orig_sr=audio_stream.rate, target_sr=target_sr)

    return audio, target_sr



# =========================================================
# Extract environmental audio (non-speech) from segment
# =========================================================
def extract_natural_audio_from_video(y, sr, start_sec, end_sec, threshold=0.3, device=None):

    # Use the device the model is already on (set at module level)
    if device is None:
        device = get_device(prefer_discrete=True)
    
    # Ensure device is a torch.device object
    if not isinstance(device, torch.device):
        device = torch.device(device)

    start = int(start_sec * sr)
    end   = int(end_sec * sr)

    audio_slice = y[start:end]

    if audio_slice.size == 0:
        return "none"

    # --- speech removal ---
    audio_tensor = torch.from_numpy(audio_slice).float()
    speech_segments = get_speech_ts(audio_tensor, silero_model, sampling_rate=sr)

    masked = audio_slice.copy()
    for seg in speech_segments:
        masked[seg["start"]:seg["end"]] = 0.0

    # --- AST with thread lock for GPU safety ---
    inputs = AST_FE(masked, sampling_rate=sr, return_tensors="pt", padding=True).to(device)

    with _ast_lock:  # Protect GPU operations from concurrent access
        with torch.no_grad():
            outputs = AST_MODEL(**inputs)
            probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

    labels = [
        AST_MODEL.config.id2label[i]
        for i, p in enumerate(probs)
        if p >= threshold
    ]
    scores = [float(p) for p in probs if p >= threshold]

    if not labels:
        return "none"

    out = [
        f"{labels[i].lower().replace('_', ' ')} (conf={scores[i]:.2f})"
        for i in range(len(labels))
    ]

    return ", ".join(out)



# =========================================================
# Main pipeline: video → scenes → natural audio labels
# =========================================================
def extract_sounds(video_path, scenes, target_sr=16000, debug=False, max_workers: int = 1):
    """
    Extract environmental audio (non-speech) from video scenes.
    
    Args:
        video_path: Path to video file
        scenes: List of scene dictionaries with start_seconds and end_seconds
        target_sr: Target sample rate
        debug: Print debug information
        max_workers: Number of parallel workers for processing scenes (1=sequential, >1=parallel)
    
    Returns:
        Updated scenes with "audio_natural" key added
    """
    # Get device for AST model (model is already on device from module load)
    device = get_device(prefer_discrete=True)

    # 1. fully decode audio using PyAV (do this once)
    y, sr = load_audio_av(video_path, target_sr)

    if max_workers == 1:
        # Sequential execution (original behavior)
        for scene in scenes:
            start = scene["start_seconds"]
            end   = scene["end_seconds"]

            label = extract_natural_audio_from_video(y, sr, start, end, device=device)
            scene["audio_natural"] = label

            if debug:
                print(f"{start} → {end} : {label}")

        return scenes
    else:
        # Parallel execution
        return _extract_sounds_parallel(
            scenes=scenes,
            audio=y,
            sr=sr,
            device=device,
            debug=debug,
            max_workers=max_workers
        )


def _extract_sounds_parallel(
    scenes: list,
    audio: np.ndarray,
    sr: int,
    device,
    debug: bool,
    max_workers: int,
):
    """
    Internal function for parallel sound extraction.
    Processes scenes in parallel using the pre-loaded audio.
    """
    
    def process_single_scene(idx, scene):
        """Process a single scene."""
        try:
            start = scene["start_seconds"]
            end   = scene["end_seconds"]

            label = extract_natural_audio_from_video(audio, sr, start, end, device=device)
            
            new_scene = dict(scene)
            new_scene["audio_natural"] = label

            if debug:
                print(f"Scene {idx} ({start} → {end}): {label}")

            return (idx, new_scene)
        except Exception as e:
            raise Exception(f"Scene {idx} failed: {str(e)}")
    
    # Pre-allocate results list to maintain order
    results = [None] * len(scenes)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all scenes
        future_to_idx = {}
        for idx, scene in enumerate(scenes):
            future = executor.submit(process_single_scene, idx, scene)
            future_to_idx[future] = idx
        
        if debug:
            print(f"\n  ⏳ Processing {len(scenes)} scenes with {max_workers} parallel workers...\n")
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_idx):
            idx, new_scene = future.result()
            results[idx] = new_scene
            completed += 1
            if debug:
                print(f"  Progress: {completed}/{len(scenes)} scenes completed")
    
    return results



# =========================================================
# Example usage
# =========================================================
def test():
    import json

    test_video = r"Videos\Watch Malala Yousafzai's Nobel Peace Prize acceptance speech.mp4"

    with open("./captioned_scenes.json", "r") as f:
        scenes = json.load(f)

    result = extract_sounds(
        test_video,
        scenes,
        debug=True
    )

# test()