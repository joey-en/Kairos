from src.debug_utils import *
from src.log_utils import *

test_videos = {
    "sponge_pyscene_blip_yolo_ASR_AST_GeminiPro25": r"Videos\SpongeBob SquarePants - Writing Essay - Some of These - Meme Source.mp4",
    "malala_pyscene_blip_yolo_ASR_AST_GeminiPro25": r"Videos\Watch Malala Yousafzai's Nobel Peace Prize acceptance speech.mp4",
    "car_pyscene_blip_yolo_ASR_AST_GeminiPro25": r"Videos\Cartastrophe.mp4",
    "spain_pyscene_blip_yolo_ASR_AST_GeminiPro25": r"Videos\Spain Vlog.mp4",
}
OUTPUT_DIR = "spain"

for OUTPUT_DIR, test_video in test_videos.items():
    log = initiate_log(video_path=test_video, run_description="Test run for video processing pipeline.")
    step = {}

    scenes, step['get_scene_list'] = get_scene_list_log(test_video, min_scene_sec=2) 
    scenes, step['save_clips'] = save_clips_log(test_video, scenes, output_dir=f"./{OUTPUT_DIR}/clips")
    see_scenes_cuts(scenes)

    scenes_with_frames, step['sample_frames'] = sample_frames_log(
        input_video_path=test_video,
        scenes=scenes,
        num_frames=3,
        new_size = 320,
        output_dir=f"./{OUTPUT_DIR}/frames",
    )

    captioned_scenes, step['caption_frames'] = caption_frames_log(
        scenes=scenes_with_frames,
        max_length=30,
        num_beams=4,
        do_sample=False,
        debug=True,
        prompt="a video frame of"
    )

    detected_obj_scenes, step['detect_object_yolo'] = detect_object_yolo_log(
        scenes= captioned_scenes,
        model_size = "model/yolov8s",
        conf = 0.5,
        iou = 0.45,
        output_dir=f"./{OUTPUT_DIR}/yolo",
    )

    sound_audio, step['ast_timings'] = extract_sounds_log(
            test_video,
            scenes=detected_obj_scenes,
            debug=True
    )

    speech_audio, step['asr_timings'] = extract_speech_log(
            video_path = test_video, 
            scenes = sound_audio, 
            model="small",
            use_vad=True, 
            target_sr=16000,
            debug = True
        )

    described_scenes, step['describe_scenes'] = describe_scenes_log(
        scenes= speech_audio,
        YOLO_key="yolo_detections",
        FLIP_key="frame_captions",
        ASR_key= "audio_natural",
        AST_key= "audio_speech",
        debug= True,
        prompt_path= "prompts/flash_scene_prompt_manahil.txt",
        model= "gemini-2.5-flash",
    )

    save_safe_df = save_vid_df(described_scenes, f"{OUTPUT_DIR}/captioned_scenes.json")
    log = complete_log(log, step, vid_len=scenes[-1]["end_seconds"], scene_num=len(scenes), vid_df= save_safe_df)
    save_log(log, filename=OUTPUT_DIR)