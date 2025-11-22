from src.debug_utils import *
from src.log_utils import *

test_video = r"Videos\SpongeBob SquarePants - Writing Essay - Some of These - Meme Source.mp4"

log = initiate_log(video_path=test_video, run_description="Test run for video processing pipeline.")
step = {}

scenes, step['get_scene_list'] = get_scene_list_log(test_video, min_scene_sec=2) 
scenes, step['save_clips'] = save_clips_log(test_video, scenes, output_dir="./output/clips")
see_scenes_cuts(scenes)

scenes_with_frames, step['sample_frames'] = sample_frames_log(
    input_video_path=test_video,
    scenes=scenes,
    num_frames=3,
    new_size = 320,
    output_dir="./output/frames",
)

captioned_scenes, step['caption_frames'] = caption_frames_log(
    scenes=scenes_with_frames,
    max_length=30,
    num_beams=4,
    do_sample=False,
    debug=True,
    prompt="a video frame of"
)
save_safe_df = save_vid_df(captioned_scenes, "output/captioned_scenes.json")

log = complete_log(log, step, vid_len=scenes[-1]["end_seconds"], scene_num=len(scenes), vid_df= save_safe_df)
save_log(log, folder="logs", filename="sponge_40_scenethreshold_justblip")