# utils.py
import cv2

def extract_frames_by_index(video_path: str, frame_indices: list[int]):
    """
    Extracts the actual frame images from the video.
    Returns list of BGR images.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))

    cap.release()
    return frames
