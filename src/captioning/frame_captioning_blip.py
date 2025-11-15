import time
from typing import List, Dict, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class BLIPCaptioner:
    """
    Wrapper around BLIP for modular use inside the fusion pipeline.
    Handles:
        - model loading
        - caption generation
        - timing measurement
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: Optional[str] = None,
    ):
        start_time = time.time()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load processor and model ONCE
        self.processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

        self.load_time = time.time() - start_time

    def _to_pil(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image)

        raise TypeError("Input image must be numpy array or PIL.Image")

    def caption(
        self,
        image: Union[np.ndarray, Image.Image],
        prompt: Optional[str] = None,
        max_length: int = 30,
        num_beams: int = 3,
        do_sample: bool = False,
    ) -> str:
        """Generate caption for a single frame."""
        pil_img = self._to_pil(image)

        inputs = (
            self.processor(pil_img, prompt, return_tensors="pt")
            if prompt
            else self.processor(pil_img, return_tensors="pt")
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2,
            )

        return self.processor.decode(output_ids[0], skip_special_tokens=True).strip()

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
        Add BLIP captions to scene frame list.
        """
        enriched = []

        for scene in scenes:
            frames = scene.get("frames", [])
            captions = []

            for frame in frames:
                t0 = time.time()
                cap = self.caption(
                    frame,
                    prompt=prompt,
                    max_length=max_length,
                    num_beams=num_beams,
                    do_sample=do_sample,
                )
                t1 = time.time() - t0

                captions.append({
                    "caption": cap,
                    "time": t1,
                })

                if debug:
                    print(f"[BLIP] {cap} (t={t1:.3f}s)")

            new_scene = dict(scene)
            new_scene["blip_captions"] = captions
            enriched.append(new_scene)

        return enriched
