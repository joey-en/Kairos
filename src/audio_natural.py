# src/audio_natural.py
import numpy as np
import torch
import librosa
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from pyannote.audio import Pipeline
import os

HF_TOKEN = os.getenv("HF_TOKEN")
vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=HF_TOKEN)

def extract_natural_audio_labels(audio_path: str, clip_sec: int = 2, device: str = "cpu", threshold: float = 0.3):
    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    
    # VAD mask speech
    try:
        vad_result = vad_pipeline(str(audio_path))
        speech_segments = [(s.start, s.end) for s in vad_result.get_timeline().support()]
    except Exception:
        speech_segments = []

    y_masked = y.copy()
    for s, e in speech_segments:
        y_masked[int(s*sr):int(e*sr)] = 0.0

    # AST model
    model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = ASTForAudioClassification.from_pretrained(model_name, use_auth_token=HF_TOKEN).to(device)
    model.eval()

    results = []
    clip_len = clip_sec * sr
    num_clips = max(1, int(np.ceil(len(y_masked)/clip_len)))

    for i in range(num_clips):
        start = int(i*clip_len)
        end = int(min((i+1)*clip_len, len(y_masked)))
        clip = y_masked[start:end]
        if clip.size == 0:
            continue

        inputs = feature_extractor(clip, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

        labels = [model.config.id2label[idx] for idx, p in enumerate(probs) if p>=threshold]
        scores = [float(p) for p in probs if p>=threshold]
        results.append({"clip_index": i, "start_sec": start/sr, "end_sec": end/sr, "labels": labels, "scores": scores})

    return results
