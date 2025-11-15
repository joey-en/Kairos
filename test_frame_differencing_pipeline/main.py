# main.py
"""
Scene-centric video analysis pipeline with full metrics logging:
1. Load video frames
2. Detect scenes
3. Detect motion frames per scene
4. Generate BLIP captions per motion frame
5. Save metrics and outputs per scene
"""

import os
import sys
import cv2
import json
from pathlib import Path
from blip_captioner import BLIPCaptioner
from scene_detector import detect_scenes
from mog2_frame_differencing import compute_mog2_differences
from metrics import MetricsLogger

# ----------------------------
# Context manager to suppress stderr (FFmpeg/OpenCV warnings only)
# ----------------------------
class SuppressStderr:
    """Context manager to suppress all stderr messages (FFmpeg/OpenCV warnings)."""
    def __enter__(self):
        self._original_stderr = sys.stderr
        self._devnull = open(os.devnull, 'w')
        sys.stderr = self._devnull
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self._original_stderr
        self._devnull.close()


def main(video_path: str, do_caption=True, output_dir="output"):
    logger = MetricsLogger()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # --------------------------
    # Load BLIP model
    # --------------------------
    print("Loading BLIP model...")
    blip = BLIPCaptioner() if do_caption else None
    logger.log_step("BLIP model loaded" if do_caption else "BLIP skipped")

    # --------------------------
    # Load video frames
    # --------------------------
    frames = []
    frame_count = 0
    with SuppressStderr():
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Loaded {frame_count} frames...")
    cap.release()
    logger.log_step(f"Frames loaded: {len(frames)}")
    print(f"Total frames loaded: {len(frames)}")

    # --------------------------
    # Detect scenes
    # --------------------------
    print("Detecting scenes...")
    with SuppressStderr():
        scenes_tuples = detect_scenes(video_path)  # returns list of (start_frame, end_frame)
    logger.log_step(f"{len(scenes_tuples)} scenes detected")
    print(f"Detected {len(scenes_tuples)} scenes.")

    # --------------------------
    # Process each scene
    # --------------------------
    scenes = []
    for scene_idx, scene_tuple in enumerate(scenes_tuples):
        start, end = scene_tuple
        scene_frames = frames[start:end+1]
        logger.log_step(f"Start processing Scene {scene_idx}")
        print(f"\nProcessing Scene {scene_idx}: frames {start}-{end}")

        # 1. Motion detection
        with SuppressStderr():
            motion_events = compute_mog2_differences(scene_frames)
        logger.log_step(f"Scene {scene_idx} motion detection done ({len(motion_events)} frames)")
        print(f"Scene {scene_idx}: {len(motion_events)} motion frames detected")

        # 2. Extract frames with motion
        motion_indices = [e['frame_index'] for e in motion_events]
        frames_with_motion = [scene_frames[idx] for idx in motion_indices]

        # 3. BLIP captioning per motion frame
        captions = []
        if do_caption and frames_with_motion:
            for i, frame in enumerate(frames_with_motion):
                caption = blip.caption_frame(frame, prompt="a video frame of")
                captions.append(caption)
                # Log every 10 frames to metrics
                if i % 10 == 0:
                    logger.log_step(f"Scene {scene_idx} caption frame {i}")
                    print(f"Scene {scene_idx}: Captioned frame {i}")
        logger.log_step(f"Scene {scene_idx} captions done ({len(captions)} frames)")

        # --------------------------
        # Save scene frames and captions
        # --------------------------
        scene_folder = output_dir / f"scene_{scene_idx}"
        frames_folder = scene_folder / "frames"
        frames_folder.mkdir(parents=True, exist_ok=True)

        captions_dict = {}
        for i, (frame, caption) in enumerate(zip(frames_with_motion, captions)):
            frame_filename = f"frame_{i}.jpg"
            frame_path = frames_folder / frame_filename
            cv2.imwrite(str(frame_path), frame)
            captions_dict[frame_filename] = caption

        captions_path = scene_folder / "captions.json"
        with open(captions_path, "w", encoding="utf-8") as f:
            json.dump(captions_dict, f, indent=2)

        scene_info = {
            "start_frame": start,
            "end_frame": end,
            "motion_events": motion_events,
            "frames_with_motion_count": len(frames_with_motion),
            "captions_count": len(captions)
        }
        scenes.append(scene_info)
        logger.log_step(f"Scene {scene_idx} processing completed")
        print(f"Scene {scene_idx}: {len(captions)} captions generated and saved")

    # --------------------------
    # Save metrics to file (UTF-8)
    # --------------------------
    metrics_path = output_dir / "pipeline_metrics.txt"
    logger.save(metrics_path)
    print(f"Metrics saved: {metrics_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = r'..\Videos\SpongeBob SquarePants - Writing Essay - Some of These - Meme Source.mp4'
    main(video_path, do_caption=True, output_dir="output")
