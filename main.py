from src.scene_cutting import get_scene_list
from src.frame_sampling import sample_frames
from src.frame_captioning_blip import caption_frames
# from src.frame_captioning_heavy import refine_caption_frames   
from src.debug_utils import *

test_video = r'Videos\SpongeBob SquarePants - Writing Essay - Some of These - Meme Source.mp4'

scenes = get_scene_list(test_video)
see_scenes_cuts(scenes)

scenes_with_frames = sample_frames(
    input_video_path=test_video,
    scenes=scenes,
    num_frames=3,
    output_dir="./output/frames",
)
# Has these errors (we need to check why):
# [h264 @ 000002602f6b6600] mmco: unref short failure
# [h264 @ 000002602fc03780] mmco: unref short failure

captioned_scenes = caption_frames(
    scenes=scenes_with_frames,
    max_length=30,
    num_beams=4,
    do_sample=False,
    debug=True,
    prompt="a video frame of"
)

save_scenes_to_file(captioned_scenes, "output/captioned_scenes.json")

# refined_scenes = refine_caption_frames(
#     scenes=captioned_scenes,
#     num_prev=1,
#     num_next=1,
#     extra_instruction=(
#         "Using the image and these captions as temporal context, "
#         "write ONE concise sentence describing what is happening "
#         "in this frame, focusing on new details or clarifications."
#     ),
#     do_sample=False,
#     debug=True,
# )

# see_first_scene(refined_scenes)