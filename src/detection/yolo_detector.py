from ultralytics import YOLO
import os
import time
import psutil

from src.perf_utils import get_gpu_stats

class YOLODetector:
    def __init__(self, model_path="yolov8s.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        t0 = time.time()
        res = self.model(frame)[0]
        t1 = time.time() - t0

        cpu = psutil.cpu_percent()
        ram = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        gpu_util, vram = get_gpu_stats()

        detections = []
        for b in res.boxes:
            cls = int(b.cls.cpu().numpy()[0])
            conf = float(b.conf.cpu().numpy()[0])
            detections.append({"label": res.names[cls], "confidence": conf})

        return detections, t1, cpu, ram, gpu_util, vram
