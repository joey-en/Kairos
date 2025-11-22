import av
import numpy as np
import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


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
def extract_natural_audio_from_video(y, sr, start_sec, end_sec, threshold=0.3, device="cpu"):

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

    # --- AST ---
    inputs = AST_FE(masked, sampling_rate=sr, return_tensors="pt", padding=True).to(device)

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
def extract_sounds(video_path, scenes, target_sr=16000, debug=False):

    # 1. fully decode audio using PyAV
    y, sr = load_audio_av(video_path, target_sr)

    # 2. loop through scenes
    for scene in scenes:
        start = scene["start_seconds"]
        end   = scene["end_seconds"]

        label = extract_natural_audio_from_video(y, sr, start, end)
        scene["audio_natural"] = label

        if debug:
            print(f"{start} → {end} : {label}")

        # ADD DEBUG TO SAVE THE AUDIO CLIPS IF NEEDED

    return scenes



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