# src/audio_asr.py
import whisper
from pathlib import Path
import tempfile
import subprocess
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr

def extract_speech_asr(video_path: str, model_name="small", energy_threshold=0.001):
    """
    Extracts speech from a video and returns transcription using Whisper.
    Applies background noise reduction and skips silent clips to reduce hallucinations.

    Parameters
    ----------
    video_path : str
        Path to the video file (e.g., MP4) or WAV.
    model_name : str
        Whisper model size: tiny, base, small, medium, large.
    energy_threshold : float
        Minimum RMS energy to attempt transcription. Clips with lower energy
        are considered silent and skipped.

    Returns
    -------
    transcript : str
        Full ASR transcript of the video's speech content, or "silent" if no speech detected.

    Notes
    -----
    RMS (Root Mean Square) energy measures average audio amplitude.
    Low RMS indicates silence or very quiet audio, which often causes ASR hallucinations.
    Noise reduction is applied using spectral gating (noisereduce) before ASR to
    remove background music or environmental noise, improving transcription accuracy.
    """
    # Extract audio to temporary WAV file
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_file = Path(tmpdir) / "audio.wav"
        cmd = f'ffmpeg -y -i "{video_path}" -ar 16000 -ac 1 -vn "{audio_file}"'
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Load audio
        y, sr = librosa.load(str(audio_file), sr=16000)

        # Check for empty audio
        if y.size == 0 or librosa.get_duration(y=y, sr=sr) < 0.1:
            return "silent"

        # Compute RMS energy
        rms = np.mean(librosa.feature.rms(y=y))
        if rms < energy_threshold:
            return "silent"

        # Reduce background noise (music, ambient sound)
        y_denoised = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.9)

        # Save denoised audio temporarily for Whisper
        denoised_file = Path(tmpdir) / "audio_denoised.wav"
        sf.write(denoised_file, y_denoised, sr)

        # Load Whisper model and transcribe
        model = whisper.load_model(model_name)
        result = model.transcribe(
            str(denoised_file),
            temperature=0,  # deterministic output
            language="en",
            task="transcribe"
        )

        return result["text"].strip()
