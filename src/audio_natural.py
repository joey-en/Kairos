# src/audio_natural.py
import torch
import librosa
import numpy as np
from pathlib import Path
import tempfile
import subprocess
from transformers import ASTFeatureExtractor, ASTForAudioClassification

def extract_natural_audio_labels_ast(video_path: str, clip_sec=5, device="cpu", hf_token=None):
    """
    Use AST (Audio Spectrogram Transformer) to get AudioSet-style labels for audio clips.

    Returns:
      List[dict], each dict has keys:
        - clip_index
        - start_sec
        - end_sec
        - labels: List[str] (AudioSet class names)
        - scores: List[float], probabilities
    """

    # 1. Extract to WAV
    sr = 16000  # AST feature extractor default sampling rate according to docs. :contentReference[oaicite:4]{index=4}  
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_file = Path(tmpdir) / "audio.wav"
        cmd = f'ffmpeg -y -i "{video_path}" -ar {sr} -ac 1 -vn "{audio_file}"'
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 2. Load audio with librosa
        y, _ = librosa.load(str(audio_file), sr=sr, mono=True)
        total_sec = len(y) / sr
        if total_sec <= 0:
            return []

        # 3. Load AST model + extractor
        model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        feature_extractor = ASTFeatureExtractor.from_pretrained(model_name, use_auth_token=hf_token)
        model = ASTForAudioClassification.from_pretrained(model_name, use_auth_token=hf_token).to(device)
        model.eval()

        # 4. Split into clips and run inference
        num_clips = max(1, int(np.ceil(total_sec / clip_sec)))
        results = []

        for i in range(num_clips):
            start = int(i * clip_sec * sr)
            end = int(min((i + 1) * clip_sec * sr, len(y)))
            clip = y[start:end]
            if clip.size == 0:
                continue

            inputs = feature_extractor(clip, sampling_rate=sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # shape (1, num_labels)

            probs = torch.sigmoid(logits)[0].cpu().numpy()  # multi-label probabilities

            # pick labels above a threshold (e.g. 0.5)
            labels = []
            scores = []
            for idx, p in enumerate(probs):
                if p > 0.5:
                    labels.append(model.config.id2label[idx])
                    scores.append(float(p))

            results.append({
                "clip_index": i,
                "start_sec": float(i * clip_sec),
                "end_sec": float(min((i + 1) * clip_sec, total_sec)),
                "labels": labels,
                "scores": scores
            })

        return results
