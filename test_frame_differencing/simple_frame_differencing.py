import cv2
import numpy as np

def compute_frame_differences(
    video_path: str,
    threshold: int = 25,
    min_change_pixels: int = 5000,
):
    """
    Simple frame differencing motion detection.

    Args:
        video_path (str): Path to input video.
        threshold (int): Pixel intensity difference threshold. 
            Industry-standard: 20-30 for typical 8-bit grayscale videos to ignore minor lighting changes.
        min_change_pixels (int): Minimum number of changed pixels to register motion. 
            Industry-standard: 5000 pixels for 720p videos; adjust for resolution.

    Returns:
        List[dict]: Each dict contains:
            {
                "frame_index": int,
                "timestamp": float,
                "changed_pixels": int,
                "motion_mask": np.ndarray (binary mask),
            }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    motion_events = []
    ret, prev_frame = cap.read()
    if not ret:
        return []

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_index = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        _, thresh_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        changed_pixels = int(np.sum(thresh_mask > 0))
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if changed_pixels >= min_change_pixels:
            motion_events.append({
                "frame_index": frame_index,
                "timestamp": timestamp,
                "changed_pixels": changed_pixels,
                "motion_mask": thresh_mask,
            })

        prev_gray = gray
        frame_index += 1

    cap.release()
    return motion_events
