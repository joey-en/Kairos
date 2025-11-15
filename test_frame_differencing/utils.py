# src/utils.py
import cv2

def extract_changed_frames(video_path: str, motion_events: list):
    """
    Given motion events, extract original frames from video.

    Args:
        video_path (str): Path to the video.
        motion_events (list): Output of compute_frame_differences or other methods.

    Returns:
        List[np.ndarray]: List of BGR frames that had motion.
    """
    if not motion_events:
        return []

    cap = cv2.VideoCapture(video_path)
    changed_frames = []

    for event in motion_events:
        frame_idx = event["frame_index"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            changed_frames.append(frame)

    cap.release()
    return changed_frames
