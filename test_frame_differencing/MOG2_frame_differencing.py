import cv2
import numpy as np

def compute_mog2_differences(
    video_path: str,
    history: int = 500,
    var_threshold: int = 16,
    detect_shadows: bool = True,
    min_change_pixels: int = 5000
):
    """
    Motion detection using OpenCV MOG2 background subtractor.

    Args:
        video_path (str): Path to input video.
        history (int): Number of frames to build background model. Industry-standard: 500.
        var_threshold (int): Variance threshold for detecting foreground pixels. Industry-standard: 16.
        detect_shadows (bool): Detect shadows (can reduce false positives in indoor/outdoor scenes).
        min_change_pixels (int): Minimum number of pixels to consider motion detected.

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

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=history, varThreshold=var_threshold, detectShadows=detect_shadows
    )
    motion_events = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = fgbg.apply(frame)
        # Remove shadows (if detect_shadows=True)
        if detect_shadows:
            _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

        changed_pixels = int(np.count_nonzero(fgmask))
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if changed_pixels >= min_change_pixels:
            motion_events.append({
                "frame_index": frame_index,
                "timestamp": timestamp,
                "changed_pixels": changed_pixels,
                "motion_mask": fgmask
            })

        frame_index += 1

    cap.release()
    return motion_events
