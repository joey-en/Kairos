import numpy as np
import torch
import librosa
import time
from transformers import ASTFeatureExtractor, ASTForAudioClassification

# Load Silero VAD once
silero_model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False
)

# Correct tuple unpacking
get_speech_ts, collect_chunks, *_ = utils

# Load AST once globally for speed
AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_FE = ASTFeatureExtractor.from_pretrained(AST_MODEL_NAME)
AST_MODEL = ASTForAudioClassification.from_pretrained(AST_MODEL_NAME)
AST_MODEL.eval()

def extract_natural_audio_labels(audio_path: str, clip_sec: int = 2, device: str = "cpu", threshold: float = 0.3, enable_logs=True):
    """
    Detect environmental/non-speech audio in a file.

    Steps:
      1. Load audio
      2. Detect speech using Silero VAD
      3. Mask speech in waveform
      4. Split into fixed-length clips
      5. Extract AST features & run AST classification
      6. Return labels & timings

    Args:
        audio_path (str): Input .wav file
        clip_sec (int): Clip length in seconds
        device (str): "cpu" or "cuda"
        threshold (float): Probability threshold for AST label
        enable_logs (bool): Print debug info

    Returns:
        results (list): List of dicts per clip with labels and scores
        timings (dict): Timing metrics including 'ast_duration_sec'
    """
    timings = {}
    t0_total = time.time()

    # 1. Load audio
    t0 = time.time()
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    timings["load_audio_sec"] = time.time() - t0

    # 2. VAD detection
    t0 = time.time()
    audio_tensor = torch.from_numpy(y).float()
    speech_segments = get_speech_ts(audio_tensor, silero_model, sampling_rate=sr)
    timings["vad_detection_sec"] = time.time() - t0

    # 3. Mask speech
    t0 = time.time()
    y_masked = y.copy()
    for seg in speech_segments:
        y_masked[seg["start"]:seg["end"]] = 0.0
    timings["mask_speech_sec"] = time.time() - t0

    # 4. Split masked audio into clips
    results = []
    clip_len = clip_sec * sr
    num_clips = max(1, int(np.ceil(len(y_masked)/clip_len)))

    for i in range(num_clips):
        start = i * clip_len
        end = min((i+1) * clip_len, len(y_masked))
        clip = y_masked[start:end]
        if clip.size == 0:
            continue

        # Feature extraction
        t0 = time.time()
        inputs = AST_FE(
            clip,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        ).to(device)
        timings.setdefault("feature_extract_sec", 0)
        timings["feature_extract_sec"] += time.time() - t0

        # AST inference
        t0 = time.time()
        with torch.no_grad():
            outputs = AST_MODEL(**inputs)
            probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()
        timings.setdefault("ast_forward_sec", 0)
        timings["ast_forward_sec"] += time.time() - t0

        labels = [AST_MODEL.config.id2label[idx] for idx, p in enumerate(probs) if p >= threshold]
        scores = [float(p) for p in probs if p >= threshold]

        results.append({
            "clip_index": i,
            "start_sec": start/sr,
            "end_sec": end/sr,
            "labels": labels,
            "scores": scores
        })

    timings["ast_duration_sec"] = time.time() - t0_total

    if enable_logs:
        print(f"[AST] Finished {audio_path}")
        for r in results:
            print(f" Clip {r['clip_index']}: {r['labels']}")
        print("Timings:", timings)

    return results, timings
