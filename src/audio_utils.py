# src/audio_utils.py
import subprocess
import os
import shutil

def get_ffmpeg_executable():
    """
    Try to locate ffmpeg automatically:
    1. Use 'ffmpeg' in PATH
    2. Check common locations
    3. Raise error if not found
    """
    # Try PATH
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    # Try common Windows install locations (winget, gyan, etc.)
    possible_paths = [
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg-*\bin\ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    import glob
    for pattern in possible_paths:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    # Not found
    raise FileNotFoundError(
        "ffmpeg executable not found. Please install FFmpeg or add it to PATH."
    )


def extract_scene_audio_ffmpeg(input_video, output_wav, start_sec, end_sec):
    """
    Extracts audio for one scene using FFmpeg.
    Auto-detects ffmpeg executable.
    """
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)

    ffmpeg_exe = get_ffmpeg_executable()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", input_video,
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-ar", "16000",
        "-ac", "1",
        "-vn",
        output_wav
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return output_wav
