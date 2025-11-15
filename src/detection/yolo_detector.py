from ultralytics import YOLO
import torch

class YOLODetector:
    def __init__(self, model_path="yolov8s.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            label = results.names[cls_id]
            conf = float(box.conf.cpu().numpy()[0])

            detections.append({
                "label": label,
                "confidence": conf
            })

        return detections
