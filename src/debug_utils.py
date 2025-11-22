import os
import subprocess
from pathlib import Path
import imageio_ffmpeg as ffmpeg

def see_first_scene(df):
    print("Printing first captioned scene:")
    print("{")
    for key in df[0]:
        if key == "frames": continue
        print(f"{key}, {df[0][key]},")
    print("}")

def see_scenes_cuts(df):
    print(f"Found {len(df)} scenes.")
    for s in df:
        print(
            f"Scene {s['scene_index']:03d}: "
            f"{s['start_timecode']} -> {s['end_timecode']} "
            f"({s['duration_seconds']:.2f} sec)"
        )

def save_clips(video_path, scenes, output_dir):
    ffmpeg_path = ffmpeg.get_ffmpeg_exe()   # portable FFmpeg binary
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    updated_scenes = []

    for scene in scenes:
        start = scene["start_seconds"]
        end = scene["end_seconds"]
        duration = end - start

        scene_index = scene.get("scene_index", len(updated_scenes))
        clip_filename = f"scene_{scene_index:04d}.mp4"
        clip_path = output_dir / clip_filename

        cmd = [
            ffmpeg_path,
            "-y",
            "-i", video_path,
            "-ss", str(start),
            "-t", str(duration),
            "-c", "copy",
            str(clip_path)
        ]

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        scene_new = dict(scene)
        scene_new["clip_path"] = str(clip_path)
        updated_scenes.append(scene_new)

    return updated_scenes

def save_vid_df(df, filepath):
    import json
    cleaned = [
        {k: v for k, v in scene.items() if k != "frames"}
        for scene in df
    ]
    with open(filepath, 'w') as f:
        json.dump(cleaned, f, indent=4)
    print(f"Scenes saved to {filepath}")
    return cleaned