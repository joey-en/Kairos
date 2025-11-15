# blip_captioner.py
from typing import List, Optional, Dict
import torch
from PIL import Image
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration

class BLIPCaptioner:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BLIPCaptioner] Loading BLIP model on {self.device}...")

        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base", use_fast=True
        )

    def caption_frame(
        self,
        frame,
        prompt: Optional[str] = None,
        max_length: int = 30,
        num_beams: int = 3,
        do_sample: bool = False,
    ) -> str:
        """
        Generate a caption for a single frame (numpy array or PIL image)
        """
        # Convert to PIL.Image in RGB
        if isinstance(frame, Image.Image):
            pil_image = frame.convert("RGB")
        elif isinstance(frame, np.ndarray):
            if frame.ndim == 3 and frame.shape[2] == 3:  # BGR -> RGB
                frame = frame[:, :, ::-1]
            pil_image = Image.fromarray(frame)
        else:
            raise TypeError("frame must be a numpy array or PIL.Image.Image")

        # Prepare inputs
        inputs = self.processor(pil_image, prompt, return_tensors="pt") if prompt else self.processor(pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate caption
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2,
            )

        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()

    def caption_scenes(
        self,
        scenes: List[Dict],
        prompt: Optional[str] = None,
        max_length: int = 30,
        num_beams: int = 3,
        do_sample: bool = False,
        debug: bool = False,
    ) -> List[Dict]:
        """
        Caption all frames in a list of scene dictionaries.
        Each scene dict should have a "frames" key containing frames.
        Returns enriched scenes with "frame_captions" key.
        """
        enriched_scenes: List[Dict] = []

        for scene in scenes:
            if debug:
                print(f"[Scene {scene.get('scene_index', '?')}]")
            frames = scene.get("frames", [])
            captions = []

            for frame in frames:
                caption = self.caption_frame(
                    frame,
                    prompt=prompt,
                    max_length=max_length,
                    num_beams=num_beams,
                    do_sample=do_sample,
                )
                captions.append(caption)
                if debug:
                    print(f"  {caption}")

            new_scene = dict(scene)
            new_scene["frame_captions"] = captions
            enriched_scenes.append(new_scene)

        return enriched_scenes
