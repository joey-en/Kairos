from src.scene_cutting import get_scene_list
from src.frame_sampling import sample_frames
from src.frame_captioning_blip import caption_frames
from src.debug_utils import *

test_video = r"Videos\Spain Vlog.mp4"

scenes = get_scene_list(test_video, min_scene_sec=2) 
scenes = save_clips(test_video, scenes, output_dir="./output/clips")
see_scenes_cuts(scenes)

scenes_with_frames = sample_frames(
    input_video_path=test_video,
    scenes=scenes,
    num_frames=3,
    new_size = 320,
    output_dir="./output/frames",
)

captioned_scenes = caption_frames(
    scenes=scenes_with_frames,
    max_length=30,
    num_beams=4,
    do_sample=False,
    debug=True,
    prompt="a video frame of"
)

save_vid_df(captioned_scenes, "output/captioned_scenes.json")
