import os
import time
import psutil
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from src.perf_utils import get_gpu_stats


class BLIPCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", device=None):
        t0 = time.time()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load processor and model ONCE
        self.processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.load_time = time.time() - t0

    def _to_pil(self, img):
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)
        raise TypeError("Unsupported image type")

    def caption(self, img, prompt=None):
        pil = self._to_pil(img)
        inputs = self.processor(pil, prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=30, num_beams=3)
        t1 = time.time() - t0

        cpu = psutil.cpu_percent() 
        ram = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        gpu_util, vram = get_gpu_stats()

        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption, t1, cpu, ram, gpu_util, vram