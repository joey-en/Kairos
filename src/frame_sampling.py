import os
from typing import List, Dict, Optional
import cv2
import numpy as np  

def resize_frame(frame, new_size=320):
    """
    Resize so the longest side = new_size, preserving aspect ratio.
    Works for vertical, horizontal, or square frames.
    """
    h, w = frame.shape[:2]
    longest = max(w, h)

    scale = new_size / longest
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized

def sample_from_clip(
    input_video_path: str,
    scene_index: int,
    start_seconds: float,
    end_seconds: float,
    num_frames: int = 5,
    new_size: int = 320,
) -> List[np.ndarray]:
    """
    Sample `num_frames` frames from a single scene interval.
    Returns ONLY the images (as numpy arrays), no saving, no dicts.

    Parameters
    ----------
    input_video_path : str
        Path to the input video file.
    scene_index : int
        Scene index (not used in logic, just for potential logging/debug).
    start_seconds : float
        Scene start time in seconds.
    end_seconds : float
        Scene end time in seconds.
    num_frames : int, default 5
        Number of frames to sample within [start_seconds, end_seconds].

    Returns
    -------
    List[np.ndarray]
        List of frames as BGR numpy arrays (OpenCV format).
        Length may be <= num_frames if decoding fails on some positions.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Convert seconds → frame indices (inclusive range)
    start_frame = int(round(start_seconds * fps))
    end_frame = int(round(end_seconds * fps)) - 1

    # Clamp to valid range
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(0, min(end_frame, total_frames - 1))

    if end_frame < start_frame:
        end_frame = start_frame

    # Compute evenly spaced positions
    if num_frames <= 1 or start_frame == end_frame:
        frame_positions = [start_frame]
    else:
        frame_positions = [
            int(round(start_frame + (i / (num_frames - 1)) * (end_frame - start_frame)))
            for i in range(num_frames)
        ]

    # Final clamp and deduplicate (just in case of rounding collisions)
    frame_positions = sorted(
        set(max(0, min(p, total_frames - 1)) for p in frame_positions)
    )

    frames: List[np.ndarray] = []

    for frame_num in frame_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret or frame is None:
            # Skip unreadable frames, but keep going
            continue
        frame_resized = resize_frame(frame, new_size)
        frames.append(frame_resized)

    cap.release()
    return frames

def sample_frames(
    input_video_path: str,
    scenes: List[Dict],
    num_frames: int = 4,
    new_size: int = 320,
    output_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Loop over a list of scene dictionaries and attach sampled frames to each.

    Parameters
    ----------
    input_video_path : str
        Path to the input video file.
    scenes : List[Dict]
        Output of get_scene_list(...), each with at least:
        - "scene_index"
        - "start_seconds"
        - "end_seconds"
    num_frames : int, default 5
        Number of frames to sample per scene.
    output_dir : Optional[str], default None
        If None  -> do NOT save frames to disk.
        If str   -> save frames under this directory (with subfolders per scene).

    Returns
    -------
    List[Dict]
        New list of scene dicts. Each scene dict is the same as input,
        plus:
            - "frames": List[np.ndarray]    (sampled images in memory)
            - "frame_paths": List[str] or None
              (paths where frames were saved, if output_dir is provided)
    """
    # Prepare saving directory if requested
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    enriched_scenes: List[Dict] = []

    for scene in scenes:
        scene_index = scene["scene_index"]
        start_seconds = scene["start_seconds"]
        end_seconds = scene["end_seconds"]

        # Use the singular helper: no dictionary involved here
        frames = sample_from_clip(
            input_video_path=input_video_path,
            scene_index=scene_index,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            num_frames=num_frames,
            new_size=new_size,
        )

        frame_paths: Optional[List[str]] = None

        # Optionally save frames if output_dir is provided
        if output_dir is not None:
            scene_folder = os.path.join(output_dir, f"scene_{scene_index:03d}")
            os.makedirs(scene_folder, exist_ok=True)

            frame_paths = []
            for idx, frame in enumerate(frames):
                filename = f"frame_{idx:02d}.jpg"
                frame_path = os.path.join(scene_folder, filename)
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)

        # Build new scene dict with frames attached
        new_scene = dict(scene)  # shallow copy
        new_scene["frames"] = frames                # in-memory images
        new_scene["frame_paths"] = frame_paths      # list of paths or None

        enriched_scenes.append(new_scene)

    return enriched_scenes
