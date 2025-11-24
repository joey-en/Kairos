"""
PARALLEL VIDEO PROCESSING PIPELINE
===================================

This script processes videos with 3 stages of parallelization:

STAGE 1 - Scene Detection (Sequential)
  └─ Detect scenes and extract frames

STAGE 2 - Vision + Audio Processing (4 Parallel Tasks)
  ├─ BLIP Frame Captioning (with internal scene-level parallelism)
  ├─ YOLO Object Detection (with internal scene-level parallelism)
  ├─ AST Sound Extraction (with internal scene-level parallelism)
  └─ ASR Speech Recognition (with internal scene-level parallelism)

STAGE 3 - Scene Description (Configurable Parallel Workers)
  └─ Gemini API calls for scene descriptions
  
Configuration:
  Stage 2 (Internal Parallelism):
  - CAPTION_FRAMES_MAX_WORKERS: BLIP scene processing (1-5 recommended)
  - DETECT_OBJECT_YOLO_MAX_WORKERS: YOLO scene processing (1-4 recommended)
  - EXTRACT_SOUNDS_MAX_WORKERS: AST scene processing (1-4 recommended)
  - EXTRACT_SPEECH_MAX_WORKERS: ASR scene processing (1-4 recommended)
    * 1 = Sequential (safest for GPU memory)
    * 2-3 = Balanced (recommended)
    * 4+ = Aggressive (faster but may use more GPU memory)
  
  Stage 3 (API Parallelism):
  - DESCRIBE_SCENES_MAX_WORKERS: Controls API call parallelism (1-5 recommended)
    * 1 = Sequential (safest, slowest)
    * 2-3 = Balanced (recommended)
    * 4-5 = Aggressive (faster but may hit rate limits)
  
  - DESCRIBE_SCENES_RATE_LIMIT: Delay between API calls per worker (seconds)
    * Higher = Safer from rate limits, slower
    * Lower = Faster but risky
"""

from src.debug_utils import *
from src.log_utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ========================================
# CONFIGURATION
# ========================================

# Videos to process
test_videos = {
    "malala_pyscene_blip_yolo_ASR_AST_GeminiPro25": r"Videos\Watch Malala Yousafzai's Nobel Peace Prize acceptance speech.mp4",
}

# Parallelization settings for Stage 2 tasks (internal scene-level parallelism)
# Adjust based on GPU memory and performance
# NOTE: Thread locks are now in place to prevent GPU memory corruption.
# If you still experience crashes, reduce workers to 1-2 for all tasks.
CAPTION_FRAMES_MAX_WORKERS = 5  # BLIP: Number of parallel scene processors (1-5 recommended)
DETECT_OBJECT_YOLO_MAX_WORKERS = 1  # YOLO: Number of parallel scene processors (1-4 recommended)
EXTRACT_SOUNDS_MAX_WORKERS = 1  # AST: Number of parallel scene processors (1-4 recommended)
EXTRACT_SPEECH_MAX_WORKERS = 1  # ASR: Number of parallel scene processors (1-4 recommended)

# Parallelization settings for describe_scenes (Stage 3)
# Adjust these to avoid API rate limits and timeouts
DESCRIBE_SCENES_MAX_WORKERS = 30  # Number of parallel API calls (1-5 recommended)
DESCRIBE_SCENES_RATE_LIMIT = 0.5  # Seconds to wait between API calls per worker

AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

def process_video_parallel(OUTPUT_DIR, test_video):
    """Process a single video with parallel execution where possible."""
    print(f"\n{'='*60}")
    print(f"Processing: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    log = initiate_log(video_path=test_video, run_description="Parallel test run for video processing pipeline.")
    step = {}
    
    # ========================================
    # STAGE 1: Sequential preprocessing
    # ========================================
    print("STAGE 1: Scene Detection")
    start_time = time.time()
    
    scenes, step['get_scene_list'] = get_scene_list_log(test_video, min_scene_sec=2)
    see_scenes_cuts(scenes)
    
    # Save clips and sample frames can run in parallel, but sample_frames is critical path
    scenes, step['save_clips'] = save_clips_log(test_video, scenes, output_dir=f"./{OUTPUT_DIR}/clips")
    
    scenes_with_frames, step['sample_frames'] = sample_frames_log(
        input_video_path=test_video,
        scenes=scenes,
        num_frames=3,
        new_size=320,
        output_dir=f"./{OUTPUT_DIR}/frames",
    )
    
    stage1_time = time.time() - start_time
    print(f"✓ Stage 1 completed in {stage1_time:.2f}s\n")
    
    # ========================================
    # STAGE 2: Parallel Processing
    # ========================================
    print("STAGE 2: Parallel Vision + Audio Processing")
    start_time = time.time()
    
    # Results storage
    results = {}
    
    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all independent tasks
        futures = {}
        
        # Vision Task 1: Frame Captioning (BLIP)
        print(f"  → Submitting: Frame Captioning (BLIP) with {CAPTION_FRAMES_MAX_WORKERS} workers")
        futures['caption_frames'] = executor.submit(
            caption_frames_log,
            scenes=scenes_with_frames,
            max_length=30,
            num_beams=4,
            do_sample=False,
            debug=True,
            prompt="a video frame of",
            max_workers=CAPTION_FRAMES_MAX_WORKERS
        )
        
        # Vision Task 2: Object Detection (YOLO)
        print(f"  → Submitting: Object Detection (YOLO) with {DETECT_OBJECT_YOLO_MAX_WORKERS} workers")
        futures['detect_object_yolo'] = executor.submit(
            detect_object_yolo_log,
            scenes=scenes_with_frames,
            model_size="model/yolov8s",
            conf=0.5,
            iou=0.45,
            output_dir=f"./{OUTPUT_DIR}/yolo",
            max_workers=DETECT_OBJECT_YOLO_MAX_WORKERS
        )
        
        # Audio Task 1: Sound Extraction (AST)
        print(f"  → Submitting: Sound Extraction (AST) with {EXTRACT_SOUNDS_MAX_WORKERS} workers")
        futures['extract_sounds'] = executor.submit(
            extract_sounds_log,
            test_video,
            scenes=scenes_with_frames,
            debug=True,
            max_workers=EXTRACT_SOUNDS_MAX_WORKERS
        )
        
        # Audio Task 2: Speech Recognition (ASR)
        print(f"  → Submitting: Speech Recognition (ASR) with {EXTRACT_SPEECH_MAX_WORKERS} workers")
        futures['extract_speech'] = executor.submit(
            extract_speech_log,
            video_path=test_video,
            scenes=scenes_with_frames,
            model="small",
            use_vad=True,
            target_sr=16000,
            debug=True,
            max_workers=EXTRACT_SPEECH_MAX_WORKERS
        )
        
        print("\n  ⏳ Running 4 tasks in parallel...\n")
        
        # Collect results as they complete
        for future_name, future in futures.items():
            try:
                result = future.result()
                results[future_name] = result
                print(f"  ✓ Completed: {future_name}")
            except Exception as e:
                print(f"  ✗ Failed: {future_name} - {str(e)}")
                raise
    
    # Merge results from parallel tasks
    captioned_scenes, step['caption_frames'] = results['caption_frames']
    detected_obj_scenes, step['detect_object_yolo'] = results['detect_object_yolo']
    sound_audio, step['ast_timings'] = results['extract_sounds']
    speech_audio, step['asr_timings'] = results['extract_speech']
    
    # Merge all data into a single scenes structure
    # Each task added data to scenes independently, so we need to combine them
    merged_scenes = merge_scene_data(
        captioned_scenes,
        detected_obj_scenes,
        sound_audio,
        speech_audio
    )
    
    stage2_time = time.time() - start_time
    print(f"\n✓ Stage 2 completed in {stage2_time:.2f}s\n")
    
    # ========================================
    # STAGE 3: Parallel Scene Description
    # ========================================
    print(f"STAGE 3: Scene Description (with {DESCRIBE_SCENES_MAX_WORKERS} parallel workers)")
    start_time = time.time()
    
    described_scenes, step['describe_scenes'] = describe_scenes_log(
        scenes=merged_scenes,
        YOLO_key="yolo_detections",
        FLIP_key="frame_captions",
        ASR_key="audio_natural",
        AST_key="audio_speech",
        debug=True,
        prompt_path="prompts/flash_scene_prompt_manahil.txt",
        model=AZURE_OPENAI_DEPLOYMENT,
        max_workers=DESCRIBE_SCENES_MAX_WORKERS,
        rate_limit_delay=DESCRIBE_SCENES_RATE_LIMIT,
    )
    
    save_safe_df = save_vid_df(described_scenes, f"{OUTPUT_DIR}/captioned_scenes.json")
    log = complete_log(log, step, vid_len=scenes[-1]["end_seconds"], scene_num=len(scenes), vid_df=save_safe_df)
    save_log(log, filename=OUTPUT_DIR)
    
    stage3_time = time.time() - start_time
    print(f"✓ Stage 3 completed in {stage3_time:.2f}s\n")
    
    print(f"{'='*60}")
    print(f"✓ Total processing time: {stage1_time + stage2_time + stage3_time:.2f}s")
    print(f"  - Stage 1 (Scene Detection):     {stage1_time:.2f}s")
    print(f"  - Stage 2 (Vision+Audio):        {stage2_time:.2f}s")
    print(f"    • BLIP: {CAPTION_FRAMES_MAX_WORKERS} workers | YOLO: {DETECT_OBJECT_YOLO_MAX_WORKERS} workers")
    print(f"    • AST: {EXTRACT_SOUNDS_MAX_WORKERS} workers | ASR: {EXTRACT_SPEECH_MAX_WORKERS} workers")
    print(f"  - Stage 3 (Scene Description):   {stage3_time:.2f}s ({DESCRIBE_SCENES_MAX_WORKERS} workers)")
    print(f"{'='*60}\n")


def merge_scene_data(*scene_lists):
    """
    Merge multiple scene lists that have different keys added.
    All scene lists should have the same number of scenes with matching timestamps.
    """
    if not scene_lists:
        return []
    
    base_scenes = scene_lists[0]
    merged = []
    
    for i, base_scene in enumerate(base_scenes):
        merged_scene = dict(base_scene)
        
        # Merge data from other scene lists
        for scene_list in scene_lists[1:]:
            if i < len(scene_list):
                # Add any keys that aren't already present
                for key, value in scene_list[i].items():
                    if key not in merged_scene:
                        merged_scene[key] = value
        
        merged.append(merged_scene)
    
    return merged


if __name__ == "__main__":
    overall_start = time.time()
    
    # Process each video sequentially (but with parallel internal processing)
    for OUTPUT_DIR, test_video in test_videos.items():
        process_video_parallel(OUTPUT_DIR, test_video)
    
    overall_time = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"ALL VIDEOS PROCESSED")
    print(f"Total time: {overall_time:.2f}s")
    print(f"{'='*60}\n")

