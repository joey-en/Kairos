# src/yolo_inference.py
from ultralytics import YOLO
import numpy as np


def run_yolo_on_frame(
    model,
    frame: np.ndarray, #process a single frame (np.ndarray)
    conf: float = 0.25,
    iou: float = 0.45,
):
    """
    Run YOLOv8 on a single frame (np.ndarray).

    Args:
        model: Loaded YOLO model object
        frame: np.ndarray frame (BGR/RGB image)
        conf: confidence threshold
        iou: IoU threshold

    Returns:
        detections: list of dictionaries with:
            {
                "label": str,
                "confidence": float,
                "bbox": [x1, y1, x2, y2]
            }
    """
    results = model.predict(
        frame,
        conf=conf,
        iou=iou,
        verbose=False
    )

    detections = []

    for r in results:
        if not hasattr(r, "boxes"):
            continue

        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf_score = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            detections.append({
                "label": label,
                "confidence": conf_score,
                "bbox": xyxy,
            })

    return detections



def detect_object_yolo(
    scenes: list, # process full scenes list
    model_size: str = "model/yolov8s.pt",
    conf: float = 0.5,
    iou: float = 0.45,
    output_dir: str = None, 
):
    """
    Run YOLO on a list of scenes.
    Adds a dict: scene["yolo_detections"] = { index: [detections], ... }

    Args:
        scenes: list of scene dictionaries
        model_size: YOLO model name (e.g., yolov8s)
        conf: confidence threshold
        iou: IoU threshold

    Returns:
        updated scenes with "yolo_detections" added
    """

    model = YOLO(model_size)

    results_scenes = []

    for s, scene in enumerate(scenes):
        new_scene = dict(scene)

        frames = scene.get("frames", [])
        yolo_dict = {}

        # process each frame in scene
        for idx, frame in enumerate(frames):
            detections = run_yolo_on_frame(
                model,
                frame,
                conf=conf,
                iou=iou
            )
            yolo_dict[idx] = detections
            if output_dir is not None:
                debug_draw_yolo(
                    frame = frame,
                    detections = detections,
                    save_path=f"./{output_dir}/scene_{s:03d}/detection_{idx:03d}.jpg",
                )

        new_scene["yolo_detections"] = yolo_dict
        results_scenes.append(new_scene)

    return results_scenes

# ================================================================
# saving images for debugging
import cv2
import os
import numpy as np
import random

# cache class → color mapping so colors stay consistent
YOLO_COLOR_MAP = {}

def get_color_for_label(label: str):
    """Return a bright, unique color for each label."""
    if label not in YOLO_COLOR_MAP:
        YOLO_COLOR_MAP[label] = (
            random.randint(80,255),
            random.randint(80,255),
            random.randint(80,255)
        )
    return YOLO_COLOR_MAP[label]


def debug_draw_yolo(
    frame: np.ndarray,
    detections: list,
    save_path: str = None
):
    """
    Draw YOLO detections on a frame for debugging.
    - Smaller text
    - Per-class consistent colors
    """

    drawn = frame.copy()

    for det in detections:
        label = det["label"]
        conf = det["confidence"]
        x1, y1, x2, y2 = map(int, det["bbox"])

        # Unique color for class
        color = get_color_for_label(label)

        # Thinner lines
        thickness = 2

        # --- Draw bounding box
        cv2.rectangle(drawn, (x1, y1), (x2, y2), color, thickness)

        # --- Smaller text
        font_scale = 0.4   # ↓ Half size from before
        font_thickness = 1

        text = f"{label} {conf:.2f}"

        (tw, th), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness
        )

        # Text background
        cv2.rectangle(
            drawn,
            (x1, y1 - th - 4),
            (x1 + tw + 2, y1),
            color,
            -1
        )

        # Text on top
        cv2.putText(
            drawn,
            text,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness
        )

    # --- save if needed
    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder != "" and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        cv2.imwrite(save_path, drawn)

    return drawn