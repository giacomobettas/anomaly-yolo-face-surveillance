import numpy as np

from src.detector import YOLODetector


def test_detector_smoke():
    """
    Smoke test for YOLODetector.

    NOTE:
    - This may download YOLO weights the first time it runs.
    - It is optional to run in day-to-day development.
    """
    det = YOLODetector(model_path="yolov8n.pt", conf_thres=0.25)

    # Dummy black image
    frame = np.zeros((480, 640, 3), dtype="uint8")
    detections = det.detect_persons(frame)

    # We don't assert on number of detections, just ensure it runs
    assert isinstance(detections, list)
