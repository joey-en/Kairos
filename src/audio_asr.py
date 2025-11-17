import torch
import librosa
import soundfile as sf
import tempfile
import time
from pathlib import Path
import noisereduce as nr
import whisper

# Load Silero VAD once (fast, CPU-friendly)
silero_model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False
)

# Correct tuple unpacking
get_speech_ts, collect_chunks, *_ = utils

# Short phrases often hallucinated by Whisper, treat them softly
HALLUCINATED = [
    "thanks", "thank you", "thanks for watching", "thank you for watching"
]

def filter_hallucinated(text: str) -> str:
    """
    Soft hallucination filter:
    Only remove text if it matches exactly a known hallucinated phrase.
    Preserves legitimate speech.
    """
    if text.strip().lower() in HALLUCINATED:
        return ""
    return text

def extract_speech_asr(audio_path: str, model_name="small", enable_logs=True):
    """
    Perform speech extraction and transcription (ASR) from audio.

    Steps:
      1. Load audio (mono 16kHz)
      2. Noise reduction
      3. Voice Activity Detection (Silero VAD)
      4. Collect speech chunks
      5. Save to temporary WAV
      6. Whisper transcription
      7. Optional hallucination filtering

    Args:
        audio_path (str): Path to input .wav
        model_name (str): Whisper model ("tiny", "base", "small", "medium", "large")
        enable_logs (bool): Print debug information

    Returns:
        text (str): Transcribed speech
        timings (dict): Detailed timing metrics:
            - load_audio_sec
            - noise_reduction_sec
            - vad_detection_sec
            - extract_chunks_sec
            - write_temp_wav_sec
            - whisper_sec
            - asr_duration_sec
    """
    timings = {}
    t0_total = time.time()

    # 1. Load audio
    t0 = time.time()
    wav, sr = librosa.load(audio_path, sr=16000, mono=True)
    timings["load_audio_sec"] = time.time() - t0

    # 2. Noise reduction
    t0 = time.time()
    wav = nr.reduce_noise(y=wav, sr=sr, prop_decrease=0.9)
    timings["noise_reduction_sec"] = time.time() - t0

    # 3. VAD
    t0 = time.time()
    wav_tensor = torch.from_numpy(wav).float()
    speech_ts = get_speech_ts(wav_tensor, silero_model, sampling_rate=sr)
    timings["vad_detection_sec"] = time.time() - t0

    if len(speech_ts) == 0:
        timings["asr_duration_sec"] = time.time() - t0_total
        if enable_logs:
            print(f"[ASR] No speech detected in {audio_path}")
        return "", timings

    # 4. Collect speech chunks
    t0 = time.time()
    # Manually combine speech segments to avoid torchcodec/collect_chunks saving
    speech_audio_segments = []
    for seg in speech_ts:
        start = int(seg["start"])
        end = int(seg["end"])
        speech_audio_segments.append(wav_tensor[start:end])
    if len(speech_audio_segments) > 0:
        speech_audio = torch.cat(speech_audio_segments).numpy()
    else:
        speech_audio = np.array([], dtype=np.float32)
    timings["extract_chunks_sec"] = time.time() - t0

    if len(speech_audio)/sr < 0.3:
        timings["asr_duration_sec"] = time.time() - t0_total
        if enable_logs:
            print(f"[ASR] Speech too short (<0.3s), skipping transcription")
        return "", timings

    # 5. Write temporary WAV
    t0 = time.time()
    tmp = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
    sf.write(tmp, speech_audio, sr)
    timings["write_temp_wav_sec"] = time.time() - t0

    # 6. Whisper transcription
    t0 = time.time()
    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(str(tmp), temperature=0, language="en")
        text = result.get("text", "").strip()
        text = filter_hallucinated(text)
    except Exception as e:
        text = ""
        if enable_logs:
            print(f"[ASR] Whisper error: {e}")
    finally:
        tmp.unlink(missing_ok=True)

    timings["whisper_sec"] = time.time() - t0
    timings["asr_duration_sec"] = time.time() - t0_total

    if enable_logs:
        print(f"[ASR] Finished {audio_path}: {len(text)} chars")
        print("Timings:", timings)

    return text, timings