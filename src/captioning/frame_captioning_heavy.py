import cv2
import numpy as np
from typing import List, Optional, Dict, Any
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer

ckpt_path = "internlm/internlm-xcomposer2-vl-7b"
device = "cuda" if torch.cuda.is_available() else "cpu"

# tokenizer + vision-language model with custom code
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    ckpt_path,
    trust_remote_code=True
).to(device).eval()

def xcomposer_frame_and_captions(
    prev_captions: List[str],
    current_caption: str,
    next_captions: List[str],
    frame_image,
    model= model,
    tokenizer= tokenizer,
    extra_instruction: str = (
        "Using the image and these captions as temporal context, "
        "write ONE concise sentence describing what is happening "
        "in this frame, focusing on new details or clarifications."
    ),
    do_sample: bool = False,
) -> str:
    """
    Use InternLM-XComposer2 to refine a single frame caption with
    surrounding context (previous & next captions).

    prev_captions : captions BEFORE this frame (older → newer order).
    current_caption : BLIP caption of this frame.
    next_captions : captions AFTER this frame (newer → future order).
    frame_image : np.ndarray (BGR or RGB), PIL.Image, or image path.
    model, tokenizer : InternLM-XComposer2 loaded with trust_remote_code=True.
    """

    # ---- Normalize image type for model.chat() ----
    if isinstance(frame_image, Image.Image):
        image_for_model = frame_image.convert("RGB")
    elif isinstance(frame_image, np.ndarray):
        # assume OpenCV BGR
        if frame_image.ndim == 3 and frame_image.shape[2] == 3:
            img_rgb = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = frame_image
        image_for_model = Image.fromarray(img_rgb)
    elif isinstance(frame_image, str):
        # assume it's a file path
        image_for_model = frame_image
    else:
        raise TypeError("frame_image must be np.ndarray, PIL.Image, or str path")

    # ---- Build textual context prompt ----
    prev_block = "\n".join(f"- {c}" for c in prev_captions) if prev_captions else "None."
    next_block = "\n".join(f"- {c}" for c in next_captions) if next_captions else "None."

    query = (
        "Previous context:\n"
        f"{prev_block}\n\n"
        "Current frame caption:\n"
        f"- {current_caption}\n\n"
        "Upcoming context:\n"
        f"{next_block}\n\n"
        "Instruction:\n"
        f"{extra_instruction}"
    )

    device = next(model.parameters()).device

    # ---- Call InternLM-XComposer2's chat API ----
    torch.set_grad_enabled(False)

    if device.type == "cuda":
        with torch.cuda.amp.autocast():
            response, _ = model.chat(
                tokenizer,
                query=query,
                image=image_for_model,
                history=[],
                do_sample=do_sample,
            )
    else:
        response, _ = model.chat(
            tokenizer,
            query=query,
            image=image_for_model,
            history=[],
            do_sample=do_sample,
        )

    return response.strip()

def refine_caption_frames(
    scenes: List[Dict],
    model = model,
    tokenizer = tokenizer,
    num_prev: int = 1,
    num_next: int = 1,
    extra_instruction: str = (
        "Using the image and these captions as temporal context, "
        "write ONE concise sentence describing what is happening "
        "in this frame, focusing on new details or clarifications."
    ),
    do_sample: bool = False,
    debug: bool = False,
) -> List[Dict]:
    """
    For each scene and each frame, call InternLM-XComposer2 with:
      - up to num_prev previous captions
      - the current BLIP caption
      - up to num_next future captions
    and attach a refined caption.

    Expects each scene dict to contain:
      - "frames": List[np.ndarray or PIL.Image]
      - "frame_captions": List[str]  (same length as frames)

    Returns a NEW list of scenes, each with:
      - "frame_detailed_captions": List[str] aligned 1:1 with frames
    """
    refined_scenes: List[Dict] = []

    for scene in scenes:
        if debug: print("Scene", scene.get("scene_index", "??"))

        frames = scene.get("frames", [])
        base_captions = scene.get("frame_captions", [])

        if len(frames) != len(base_captions):
            raise ValueError(
                f"Scene {scene.get('scene_index', '?')} has "
                f"{len(frames)} frames but {len(base_captions)} captions."
            )

        n = len(frames)
        frame_detailed_captions: List[str] = []

        for i in range(n):
            # ---- build sliding window context ----
            start_prev = max(0, i - num_prev)
            end_prev = i  # exclusive of current
            prev_captions = base_captions[start_prev:end_prev]

            current_caption = base_captions[i]

            start_next = i + 1
            end_next = min(n, i + 1 + num_next)
            next_captions = base_captions[start_next:end_next]

            frame_image = frames[i]

            refined_caption = xcomposer_frame_and_captions(
                prev_captions=prev_captions,
                current_caption=current_caption,
                next_captions=next_captions,
                frame_image=frame_image,
                model=model,
                tokenizer=tokenizer,
                extra_instruction=extra_instruction,
                do_sample=do_sample,
            )

            frame_detailed_captions.append(refined_caption)
            if debug: print(f"  {refined_caption}")

        new_scene = dict(scene)  # shallow copy
        new_scene["frame_detailed_captions"] = frame_detailed_captions
        refined_scenes.append(new_scene)

    return refined_scenes
