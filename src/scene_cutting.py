from typing import List, Dict
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

def get_scene_list(input_video_path: str, threshold: float = 27.0, min_scene_len: int = 15) -> List[Dict]:
    video = open_video(input_video_path)

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))

    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    result = []
    for idx, (start_time, end_time) in enumerate(scene_list):
        start_sec = start_time.get_seconds()
        end_sec = end_time.get_seconds()
        result.append({
            "scene_index": idx,
            "start_timecode": str(start_time),
            "end_timecode": str(end_time),
            "start_seconds": start_sec,
            "end_seconds": end_sec,
            "duration_seconds": end_sec - start_sec,
        })
    return result


def test():
    test_video = r'Videos\SpongeBob SquarePants - Writing Essay - Some of These - Meme Source.mp4'
    scenes = get_scene_list(test_video)

    print(f"Found {len(scenes)} scenes.")
    for s in scenes:
        print(
            f"Scene {s['scene_index']:03d}: "
            f"{s['start_timecode']} -> {s['end_timecode']} "
            f"({s['duration_seconds']:.2f} sec)"
        )
# test()