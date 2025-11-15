import cv2
import numpy as np
from typing import List, Union, Dict

def compute_mog2_differences(
    frames_or_video: Union[str, List[np.ndarray]],
    min_change_pixels: int = 5000,
    history: int = 500,
    var_threshold: int = 16,
    detect_shadows: bool = True,
) -> List[Dict]:
    """
    Runs MOG2 motion detection either over a full video file
    or over a list of pre-loaded frames (scene frames).

    Parameters
    ----------
    frames_or_video : str | list of np.ndarray
        Path to video file or list of frames.
    min_change_pixels : int
        Minimum number of changed pixels to count as motion.
    history : int
        History length for MOG2.
    var_threshold : int
        Variance threshold for MOG2.
    detect_shadows : bool
        Whether to detect shadows in MOG2.

    Returns
    -------
    events : list of dict
        Each dict contains:
            frame_index : int
            timestamp : float
            changed_pixels : int
    """
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows
    )

    events = []
    frame_idx = 0

    # If input is a video path
    if isinstance(frames_or_video, str):
        cap = cv2.VideoCapture(frames_or_video)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {frames_or_video}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fgmask = fgbg.apply(frame)

            if detect_shadows:
                _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

            changed_pixels = int(np.count_nonzero(fgmask))
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            if changed_pixels >= min_change_pixels:
                events.append({
                    "frame_index": frame_idx,
                    "timestamp": timestamp,
                    "changed_pixels": changed_pixels
                })

            frame_idx += 1

        cap.release()

    # If input is a list of frames
    else:
        for frame in frames_or_video:
            fgmask = fgbg.apply(frame)

            if detect_shadows:
                _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

            changed_pixels = int(np.count_nonzero(fgmask))
            # We cannot get real timestamp here; use frame index
            timestamp = frame_idx / 30.0  # assuming 30fps if needed

            if changed_pixels >= min_change_pixels:
                events.append({
                    "frame_index": frame_idx,
                    "timestamp": timestamp,
                    "changed_pixels": changed_pixels
                })

            frame_idx += 1

    return events
