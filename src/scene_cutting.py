from typing import List, Dict
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
import cv2

def get_scene_list(input_video_path: str, threshold: float = 40, min_scene_sec: int = 2, frame_skip:int=3) -> List[Dict]:
    """
    Detect scenes in a video using PySceneDetect and return structured metadata.

    Parameters
    ----------
    input_video_path : str
        Path to the input video file.
    threshold : float, optional
        Sensitivity for the ContentDetector. Lower values detect more scene cuts.
        Default is 27.0.
    min_scene_len : int, optional
        Minimum scene length in frames. Default is 15.

    Returns
    -------
    List[Dict]
        A list of dictionaries, each containing:
        - "scene_index": Index of the detected scene.
        - "start_timecode": Start timecode (HH:MM:SS.mmm).
        - "end_timecode": End timecode (HH:MM:SS.mmm).
        - "start_seconds": Start time in seconds (float).
        - "end_seconds": End time in seconds (float).
        - "duration_seconds": Duration of the scene in seconds.

    Notes
    -----
    This function uses PySceneDetect's ContentDetector to locate abrupt content
    changes. It is suitable for preprocessing steps in segmentation, retrieval,
    summarization, and other video analysis workflows.
    """
    video = open_video(input_video_path)

    # Getting the min_scene_len based on fps
    fps = cv2.VideoCapture(input_video_path).get(cv2.CAP_PROP_FPS)
    min_scene_len = int(fps * min_scene_sec) 
    
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))

    scene_manager.detect_scenes(video, frame_skip=frame_skip)
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