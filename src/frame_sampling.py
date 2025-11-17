import os
import cv2
import numpy as np

def sample_from_clip(video_path, scene_idx, start_s, end_s, n=4, margin_ratio=0.10):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Convert time to frames
    start = int(start_s * fps)
    end = max(int(end_s * fps) - 1, start)

    scene_length = end - start

    if scene_length <= 2:
        safe_start = start
        safe_end = end
    else:
        # Safety margins: 10% from start and end
        safe_start = start + int(scene_length * margin_ratio)
        safe_end = end - int(scene_length * margin_ratio)

        # Ensure valid
        if safe_start >= safe_end:
            safe_start = start
            safe_end = end

    # Sample evenly *inside the safe zone*
    positions = list(sorted(set(np.linspace(safe_start, safe_end, n, dtype=int))))

    frames = []
    for p in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, p)
        ok, f = cap.read()
        if ok:
            frames.append(f)

    cap.release()
    return frames


def sample_frames(video_path, scenes, num_frames=4, output_dir=None):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    enriched = []
    for sc in scenes:

        frames = sample_from_clip(
            video_path,
            sc["scene_index"],
            sc["start_seconds"],
            sc["end_seconds"],
            n=num_frames
        )

        frame_paths = []
        if output_dir:
            for i, fr in enumerate(frames):
                out_path = os.path.join(
                    output_dir,
                    f"scene{sc['scene_index']}_frame{i}.jpg"
                )
                cv2.imwrite(out_path, fr)
                frame_paths.append(out_path)

        nsc = dict(sc)
        nsc["frames"] = frames
        nsc["frame_paths"] = frame_paths
        enriched.append(nsc)

    return enriched
