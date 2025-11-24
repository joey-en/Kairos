from typing import List, Dict
import cv2
import numpy as np
import torch
from typing import Optional
from PIL import Image
from src.gpu_utils import get_device
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

device = get_device(prefer_discrete=True)

# ======================================================================
# Load BLIP model and processor
from transformers import BlipProcessor, BlipForConditionalGeneration
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base", use_fast=True)

# Thread lock for GPU operations to prevent memory corruption
_blip_lock = threading.Lock()
# ======================================================================
# # Load BLIP2 model and processor
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-flan-t5-xl",
#     torch_dtype=torch.float32,    # CPU-friendly
# ).to(device)

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
# ======================================================================


def blip_frame(
    image,
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    prompt: Optional[str] = None,
    max_length: int = 30,
    num_beams: int = 3,
    do_sample: bool = False,
) -> str:
    """
    Generate a BLIP caption for a single frame.

    Parameters
    ----------
    image :
        Either a NumPy array (OpenCV BGR or RGB) or a PIL.Image.
    model : BlipForConditionalGeneration
        Preloaded BLIP captioning model.
    processor : BlipProcessor
        Matching BLIP processor.
    prompt : str, optional
        Optional conditioning text, e.g. "a cartoon frame of".
        If None, uses unconditional captioning.
    max_length : int
        Maximum length of the generated caption (tokens).
    num_beams : int
        Beam search width (higher = better but slower).
    do_sample : bool
        Whether to sample (True) or keep decoding deterministic (False).

    Returns
    -------
    str
        Generated caption.
    """
    # --- Normalize image to RGB PIL.Image ---
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        # Assume OpenCV BGR by default
        if image.ndim == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        pil_image = Image.fromarray(image_rgb)
    else:
        raise TypeError("image must be a PIL.Image.Image or a numpy.ndarray")

    # Figure out model device (cpu / cuda / mps)
    device = next(model.parameters()).device

    # --- Prepare inputs for BLIP ---
    if prompt is not None:
        inputs = processor(
            pil_image,
            prompt,
            return_tensors="pt",
        )
    else:
        inputs = processor(
            pil_image,
            return_tensors="pt",
        )

    # Move tensors to same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # --- Generate caption with thread lock for GPU safety ---
    with _blip_lock:  # Protect GPU operations from concurrent access
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2,
            )

        caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


def caption_frames(
    scenes: List[Dict],
    model: BlipForConditionalGeneration = model,
    processor: BlipProcessor = processor,
    prompt: Optional[str] = None,
    max_length: int = 30,
    num_beams: int = 3,
    do_sample: bool = False,
    debug: bool = False,
    max_workers: int = 1,
) -> List[Dict]:
    """
    For each scene in `scenes`, run BLIP on each frame and attach captions.

    Parameters
    ----------
    scenes : List[Dict]
        Scene dictionaries. Each scene is expected to contain a "frames" key
        with a list of images (numpy arrays or PIL images).
    model : BlipForConditionalGeneration
        Preloaded BLIP captioning model.
    processor : BlipProcessor
        Matching BLIP processor.
    prompt : str, optional
        Optional conditioning text for all captions (e.g. "a cartoon frame of").
    max_length : int
        Max caption length (tokens).
    num_beams : int
        Beam search width.
    do_sample : bool
        Whether to sample or keep it deterministic.
    debug : bool
        Print debug information.
    max_workers : int
        Number of parallel workers for processing scenes (1=sequential, >1=parallel).

    Returns
    -------
    List[Dict]
        New list of scenes; each scene dict has an extra key:
            "frame_captions": List[str]
        aligned 1:1 with the "frames" list.
    """
    if max_workers == 1:
        # Sequential execution (original behavior)
        enriched_scenes: List[Dict] = []

        for scene in scenes:
            if debug: print("Scene", scene.get("scene_index", "??"))
            frames = scene.get("frames", [])
            captions: List[str] = []

            for frame in frames:
                caption = blip_frame(
                    image=frame,
                    model=model,
                    processor=processor,
                    prompt=prompt,
                    max_length=max_length,
                    num_beams=num_beams,
                    do_sample=do_sample,
                )
                captions.append(caption)
                if debug: print(f"  {caption}")

            new_scene = dict(scene)  # shallow copy so we don't mutate original reference
            new_scene["frame_captions"] = captions
            enriched_scenes.append(new_scene)

        return enriched_scenes
    else:
        # Parallel execution
        return _caption_frames_parallel(
            scenes=scenes,
            model=model,
            processor=processor,
            prompt=prompt,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            debug=debug,
            max_workers=max_workers
        )


def _caption_frames_parallel(
    scenes: List[Dict],
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    prompt: Optional[str],
    max_length: int,
    num_beams: int,
    do_sample: bool,
    debug: bool,
    max_workers: int,
) -> List[Dict]:
    """
    Internal function for parallel frame captioning.
    Processes scenes in parallel, with each scene processing its frames sequentially.
    PyTorch models are thread-safe for inference, so we can share the model across threads.
    """
    
    def process_single_scene(idx, scene):
        """Process a single scene with all its frames."""
        try:
            if debug:
                print(f"Processing Scene {scene.get('scene_index', idx)}")
            
            frames = scene.get("frames", [])
            captions: List[str] = []

            for frame in frames:
                caption = blip_frame(
                    image=frame,
                    model=model,  # Shared model instance (thread-safe for inference)
                    processor=processor,  # Shared processor (thread-safe)
                    prompt=prompt,
                    max_length=max_length,
                    num_beams=num_beams,
                    do_sample=do_sample,
                )
                captions.append(caption)
                if debug:
                    print(f"  Scene {scene.get('scene_index', idx)}: {caption}")

            new_scene = dict(scene)
            new_scene["frame_captions"] = captions
            
            if debug:
                print(f"✓ Scene {scene.get('scene_index', idx)} completed")
            
            return (idx, new_scene)
        except Exception as e:
            if debug:
                print(f"✗ Scene {scene.get('scene_index', idx)} failed: {str(e)}")
            raise
    
    # Pre-allocate results list to maintain order
    results = [None] * len(scenes)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all scenes
        future_to_idx = {}
        for idx, scene in enumerate(scenes):
            future = executor.submit(process_single_scene, idx, scene)
            future_to_idx[future] = idx
        
        if debug:
            print(f"\n  ⏳ Processing {len(scenes)} scenes with {max_workers} parallel workers...\n")
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_idx):
            idx, new_scene = future.result()
            results[idx] = new_scene
            completed += 1
            if debug:
                print(f"  Progress: {completed}/{len(scenes)} scenes completed")
    
    return results


'''
captioned_scenes = caption_frames(
    scenes=scenes_with_frames,
    max_length=30,
    num_beams=4,
    do_sample=False,
    debug=True,
    prompt="a video frame of"
)

Scene 0
  a video frame of a room with a blue door and a pink flower on the floor
  a video frame of a sponge sponge and his friend
  a video frame of a sponge sponge and his friend
  a video frame of a sponge sponge and his friend
Scene 1
  a video frame of sponge spongenan ' s revenge
  a video frame of sponge spongenan ' s revenge
  a video frame of sponge spongenan ' s revenge
  a video frame of sponge spongenan ' s revenge
Scene 2
  a video frame of blue and green stripes
  a video frame of a cartoon character holding a sword
  a video frame of a bottle of beer
  a video frame of a man with a hat on his head
Scene 3
  a video frame of a piece of paper on a table
  a video frame of a person writing on a piece of paper
  a video frame of an airplane flying through the sky
  a video frame of a pencil on a piece of paper
Scene 4
  a video frame of a cartoon character sitting on a chair
  a video frame of a sponge sponge and his pencil
  a video frame of a cartoon character
  a video frame of a cartoon character with blue eyes and a smile
Scene 5
  a video frame of an airplane flying in the sky
  a video frame of a hand holding a pencil
  a video frame of a cartoon character holding a pencil
  a video frame of a cartoon character holding a pencil
Scene 6
  a video frame of a cartoon character sitting at a table
  a video frame of a sponge sponge with a piece of paper
  a video frame of a sponge sponge and his friend
  a video frame of a sponge sponge with a piece of paper
Scene 7
  a video frame of a cartoon character holding a piece of paper
  a video frame of a cartoon character holding a piece of paper
  a video frame of a cartoon character holding a piece of paper
  a video frame of a cartoon character holding a piece of paper
Scene 8
  a video frame of a sponge sponge and his friend
  a video frame of sponge spongenan ' s revenge
  a video frame of a sponge sponge and his friend
  a video frame of a sponge sponge and his friend
Scene 9
  a video frame of a black background with a white border
'''