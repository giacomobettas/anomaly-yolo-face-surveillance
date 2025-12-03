"""
Video I/O utilities for the anomaly-yolo-face-surveillance project.

- Opening video files for reading
- Creating video writers for annotated outputs
- Drawing detections, labels, and reconstruction errors on frames
"""

from typing import Tuple, List, Optional

import cv2
import numpy as np


def open_video_capture(video_path: str) -> Tuple[cv2.VideoCapture, float, Tuple[int, int]]:
    """
    Open a video file for reading.

    Args:
        video_path: Path to input video.

    Returns:
        cap: cv2.VideoCapture object
        fps: frames per second (float, default 25.0 if unknown)
        frame_size: (width, height) in pixels
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # sensible default

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        # fallback values if metadata is missing
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            raise RuntimeError(f"Could not read any frame from video: {video_path}")
        height, width = frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

    return cap, fps, (width, height)


def create_video_writer(
    output_path: str,
    fps: float,
    frame_size: Tuple[int, int],
) -> cv2.VideoWriter:
    """
    Create a VideoWriter for saving annotated video.

    Args:
        output_path: Path to output video file.
        fps: Frames per second.
        frame_size: (width, height) in pixels.

    Returns:
        cv2.VideoWriter object.
    """
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    width, height = frame_size
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer for: {output_path}")
    return writer


def draw_detections(
    frame: np.ndarray,
    detections: List[dict],
    identities: Optional[List[str]] = None,
    errors: Optional[List[Optional[float]]] = None,
) -> np.ndarray:
    """
    Draw bounding boxes, labels, and reconstruction errors on the frame.

    Args:
        frame: Original BGR frame (modified in-place).
        detections: List of dicts produced by YOLODetector.detect_persons().
        identities: Optional list of identity labels, same length as detections.
        errors: Optional list of reconstruction errors (floats or None).

    Returns:
        The annotated frame (same object as input).
    """
    if identities is None:
        identities = ["person"] * len(detections)
    if errors is None:
        errors = [None] * len(detections)

    for det, identity, err in zip(detections, identities, errors):
        x1, y1, x2, y2 = det["bbox"]
        conf = det.get("conf", 0.0)

        # Convert to int for drawing
        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])

        # Bounding box
        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)

        # Label text
        label = f"{identity} ({conf:.2f})"
        if err is not None:
            label += f" err={err:.4f}"

        # Text box
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            frame,
            (x1i, y1i - th - baseline - 4),
            (x1i + tw + 4, y1i),
            (0, 255, 0),
            thickness=-1,
        )
        cv2.putText(
            frame,
            label,
            (x1i + 2, y1i - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return frame
