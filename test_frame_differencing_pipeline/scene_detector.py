# scene_detector.py
from scenedetect import SceneManager, open_video, ContentDetector

def detect_scenes(video_path: str):
    """
    Returns list of (start_frame, end_frame) scene boundaries.
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))
    scene_manager.detect_scenes(video)

    scene_list = scene_manager.get_scene_list()
    frame_scenes = []

    for start, end in scene_list:
        frame_scenes.append((start.get_frames(), end.get_frames()))

    return frame_scenes
