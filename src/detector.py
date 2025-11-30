"""
YOLOv8-based person detector using the Ultralytics library.

This module wraps a YOLO model and exposes a clean API that returns
person detections (bounding boxes and confidence scores) from a frame.
"""

from typing import List, Dict, Any, Tuple

import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """
    Simple wrapper around Ultralytics YOLO models for person detection.

    Expected input:
        - frame: numpy array (H, W, 3), BGR (as from OpenCV) or RGB.
          Ultralytics handles both, but typically we pass OpenCV frames.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        device: str = "cpu",
    ):
        """
        Args:
            model_path: Path or name of YOLO model (e.g. 'yolov8n.pt').
            conf_thres: Confidence threshold.
            iou_thres: IoU threshold for NMS.
            device: 'cpu' or 'cuda' (if available).
        """
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device

    def detect_persons(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run person detection on a single frame.

        Returns:
            A list of dicts with keys:
                - 'bbox': (x1, y1, x2, y2)
                - 'conf': float confidence score
                - 'class_id': int (should be 0 for 'person' in COCO)
        """
        if frame is None:
            return []

        # Ultralytics will infer device; we can hint via model.to() if needed.
        results = self.model(
            frame,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []

        for r in results:
            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()  # (N, 4)
            confs = r.boxes.conf.cpu().numpy()  # (N,)
            cls_ids = r.boxes.cls.cpu().numpy()  # (N,)

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                # YOLO COCO class 0 is 'person'
                if int(cls_id) != 0:
                    continue

                x1, y1, x2, y2 = box.astype(float).tolist()
                detections.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "conf": float(conf),
                        "class_id": int(cls_id),
                    }
                )

        return detections
