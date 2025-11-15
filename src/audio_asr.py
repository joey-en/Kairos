# src/audio_asr.py
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
import whisper
from pyannote.audio import Pipeline
import os

HF_TOKEN = os.getenv("HF_TOKEN")
vad_pipeline = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection", use_auth_token=HF_TOKEN
)

# List of phrases to ignore if Whisper hallucinates them
HALLUCINATED_PHRASES = [
    "thanks for watching",
    "thank you for watching",
    "thanks",
    "thank you"
]

def extract_speech_asr(audio_path: str, model_name: str = "small", min_seg_sec: float = 0.05, silence_rms_thresh: float = 0.05):
    """
    Extract speech from audio with VAD + Whisper.
    
    Skips very short segments, low energy (silent) clips, and hallucinated phrases.
    """
    # Load audio
    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    # Denoise
    y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.9)

    # Save temp WAV
    tmp_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
    sf.write(str(tmp_path), y, sr)

    # -----------------------
    # 1. VAD
    # -----------------------
    try:
        vad_result = vad_pipeline(str(tmp_path))
        speech_segments = [(s.start, s.end) for s in vad_result.get_timeline().support()]
    except Exception:
        speech_segments = [(0, len(y)/sr)]

    # Filter very short segments
    speech_segments = [(s, e) for s, e in speech_segments if e - s >= min_seg_sec]
    if len(speech_segments) == 0:
        tmp_path.unlink(missing_ok=True)
        return ""  # No speech to transcribe

    # Concatenate speech segments
    speech_audio = np.concatenate([y[int(s*sr):int(e*sr)] for s, e in speech_segments])

    # Check RMS to filter silence
    rms = np.sqrt(np.mean(speech_audio**2))
    if rms < silence_rms_thresh:
        tmp_path.unlink(missing_ok=True)
        return ""  # Mostly silence, skip transcription

    # Save filtered speech audio
    sf.write(str(tmp_path), speech_audio, sr)

    # -----------------------
    # 2. Whisper ASR
    # -----------------------
    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(str(tmp_path), temperature=0, language="en", task="transcribe")
        text = result.get("text", "").strip()
        # Remove hallucinated phrases
        lower_text = text.lower()
        if any(p in lower_text for p in HALLUCINATED_PHRASES):
            text = ""
    except Exception:
        text = ""
    finally:
        tmp_path.unlink(missing_ok=True)

    return text
