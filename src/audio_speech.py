import numpy as np
import librosa
import noisereduce as nr
import torch
import av
import whisper
from src.gpu_utils import get_device
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Thread lock for Whisper GPU operations to prevent memory corruption
_whisper_lock = threading.Lock()

# Load Silero VAD (once)
silero_model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False
)
get_speech_ts, *_ = utils


# -------------------------------------------------------
# 1. Extract audio from video (your function – cleaned)
# -------------------------------------------------------

def load_audio_av(video_path, target_sr=16000):
    container = av.open(
        video_path,
        options={"fflags": "+genpts", "ignore_editlist": "1"}
    )

    audio_stream = next(s for s in container.streams if s.type == "audio")
    audio_stream.thread_type = "AUTO"

    samples = []

    for frame in container.decode(audio_stream):
        pcm = frame.to_ndarray()           # (channels, samples)
        pcm = pcm.mean(axis=0)             # stereo → mono
        samples.append(pcm)

    if not samples:
        return np.zeros(1, dtype=np.float32), target_sr

    audio = np.concatenate(samples).astype(np.float32)

    # Resample using librosa
    audio = librosa.resample(audio, orig_sr=audio_stream.rate, target_sr=target_sr)

    return audio, target_sr


# -------------------------------------------------------
# 2. Soft VAD Speech Enhancement (preserves timestamps)
# -------------------------------------------------------

def enhance_with_soft_vad(audio, sr):
    """
    Uses Silero VAD to detect speech timestamps,
    then lightly applies denoising to ONLY speech scenes.
    Does NOT cut/remove silence → timestamps stay intact.
    """
    audio_t = torch.from_numpy(audio).float()

    speech_ts = get_speech_ts(audio_t, silero_model, sampling_rate=sr)

    if len(speech_ts) == 0:
        return audio  # no speech detected → return unchanged

    enhanced = audio.copy()

    for seg in speech_ts:
        start, end = seg["start"], seg["end"]
        segment = enhanced[start:end]

        # light denoise only on speech part
        enhanced[start:end] = nr.reduce_noise(
            y=segment,
            sr=sr,
            prop_decrease=0.7
        )

    return enhanced


# -------------------------------------------------------
# 3. Clean Audio Pipeline (extract + denoise + soft VAD)
# -------------------------------------------------------

def load_and_clean_audio(video_path, target_sr=16000, use_vad=True):
    # 1. Extract raw audio
    audio, sr = load_audio_av(video_path, target_sr)

    # 2. Full-track noise reduction
    audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.9)

    # 3. Optional soft VAD enhancement
    if use_vad:
        audio = enhance_with_soft_vad(audio, sr)

    return audio, sr


# -------------------------------------------------------
# 4. Slice Audio Based on Timestamps
# -------------------------------------------------------

def slice_audio(clean_audio, sr, t0, t1):
    """
    Slice using absolute timestamps — thanks to soft VAD,
    timestamps remain valid.
    """
    i0 = int(t0 * sr)
    i1 = int(t1 * sr)

    i0 = max(i0, 0)
    i1 = min(i1, len(clean_audio))

    return clean_audio[i0:i1]

# =========================================================
# batch
# =========================================================
def extract_speech(video_path, scenes, model, use_vad=True, target_sr=16000, debug=False, max_workers: int = 1):
    """
    Batch process timestamped audio chunks from a video.
    - Loads and cleans audio ONCE.
    - Slices audio per segment.
    - Appends "audio_speech" (numpy array) to each dict.

    Args:
        video_path (str): Path to the input video.
        scenes (list[dict]): List of dicts containing:
            - start_seconds
            - end_seconds
        model: Whisper model name (e.g., "small")
        use_vad (bool): Whether to apply soft VAD enhancement.
        target_sr (int): Sampling rate.
        debug (bool): Print debug information.
        max_workers (int): Number of parallel workers for processing scenes (1=sequential, >1=parallel)

    Returns:
        list[dict]: same list, but each entry gains key "audio_speech".
    """
    # Get device for Whisper
    device = get_device(prefer_discrete=True)
    device_str = str(device) if isinstance(device, torch.device) else device
    whisper_model = whisper.load_model(model, device=device_str)

    # 1) Load & clean audio ONCE
    clean_audio, sr = load_and_clean_audio(
        video_path,
        target_sr=target_sr,
        use_vad=use_vad,
    )

    if max_workers == 1:
        # Sequential execution (original behavior)
        for scene in scenes:
            t0 = float(scene["start_seconds"])
            t1 = float(scene["end_seconds"])

            audio_clip = slice_audio(clean_audio, sr, t0, t1)
            
            # Protect GPU operations with thread lock (even in sequential mode for safety)
            with _whisper_lock:
                speech = whisper_model.transcribe(audio_clip, fp16=False)['text']

            # add new key
            scene["audio_speech"] = speech
            if debug: print(f"Scene {t0:.2f}-{t1:.2f}s: {speech}")

        return scenes
    else:
        # Parallel execution
        return _extract_speech_parallel(
            scenes=scenes,
            clean_audio=clean_audio,
            sr=sr,
            whisper_model=whisper_model,
            debug=debug,
            max_workers=max_workers
        )


def _extract_speech_parallel(
    scenes: list,
    clean_audio: np.ndarray,
    sr: int,
    whisper_model,
    debug: bool,
    max_workers: int,
):
    """
    Internal function for parallel speech extraction.
    Processes scenes in parallel using the pre-loaded and cleaned audio.
    """
    
    def process_single_scene(idx, scene):
        """Process a single scene."""
        try:
            t0 = float(scene["start_seconds"])
            t1 = float(scene["end_seconds"])

            audio_clip = slice_audio(clean_audio, sr, t0, t1)
            
            # Protect GPU operations with thread lock
            with _whisper_lock:
                speech = whisper_model.transcribe(audio_clip, fp16=False)['text']

            new_scene = dict(scene)
            new_scene["audio_speech"] = speech

            if debug:
                print(f"Scene {idx} ({t0:.2f}-{t1:.2f}s): {speech}")

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

    result = extract_speech(
        video_path = test_video, 
        scenes = scenes, 
        model="small",
        use_vad=True, 
        target_sr=16000,
        debug = True
    )

# test()

