"""
Face encoding and identification using the face_recognition library.

We assume a directory structure like:

data/faces/
  person1/
    img1.jpg
    img2.jpg
  person2/
    img1.jpg

Each subfolder name is treated as a label (identity).
"""

import os
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

try:
    import face_recognition  # type: ignore
    _FACE_LIB_AVAILABLE = True
    _FACE_IMPORT_ERROR = None
except ImportError as e:
    face_recognition = None  # type: ignore
    _FACE_LIB_AVAILABLE = False
    _FACE_IMPORT_ERROR = e


def load_face_encodings(faces_root: str) -> Dict[str, List[np.ndarray]]:
    """
    Load face encodings from a directory of labeled face images.

    Args:
        faces_root: Root directory, e.g. data/faces.

    Returns:
        A dict mapping label -> list of 128-d encodings.
        Images without detectable faces are skipped.
    """
    if not _FACE_LIB_AVAILABLE:
        raise RuntimeError(
            "face_recognition library is not available. "
            "Install it to use face-based identification."
        )

    encodings: Dict[str, List[np.ndarray]] = {}

    if not os.path.isdir(faces_root):
        raise FileNotFoundError(f"Faces directory not found: {faces_root}")

    for label in sorted(os.listdir(faces_root)):
        label_dir = os.path.join(faces_root, label)
        if not os.path.isdir(label_dir):
            continue

        label_encs: List[np.ndarray] = []

        for fname in sorted(os.listdir(label_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(label_dir, fname)
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(img_rgb, model="hog")  # type: ignore[arg-type]
            if not boxes:
                continue

            # use the first detected face
            encs = face_recognition.face_encodings(img_rgb, known_face_locations=boxes)  # type: ignore[arg-type]
            if encs:
                label_encs.append(encs[0])

        if label_encs:
            encodings[label] = label_encs

    return encodings


def identify_face(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    known_encodings: Dict[str, List[np.ndarray]],
    tolerance: float = 0.6,
) -> Tuple[str, Optional[float]]:
    """
    Identify the face within the given bounding box using known encodings.

    Args:
        frame: Full frame (BGR or RGB).
        bbox: (x1, y1, x2, y2) person bounding box (float).
        known_encodings: dict mapping label -> list of encodings.
        tolerance: max distance to consider a match.

    Returns:
        (label, distance) where label is one of known labels or "unknown".
        distance is the best (smallest) face distance, or None if unknown.
    """
    if not _FACE_LIB_AVAILABLE:
        # Gracefully degrade: no face library -> always unknown
        return "unknown", None

    if frame is None or not known_encodings:
        return "unknown", None

    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    # Clamp bbox to frame boundaries
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return "unknown", None

    # Crop the full person bounding box; the face (if visible) should be inside it.
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown", None

    # Convert BGR->RGB if needed (face_recognition expects RGB)
    if crop.shape[-1] == 3:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    else:
        crop_rgb = crop

    boxes = face_recognition.face_locations(crop_rgb, model="hog")  # type: ignore[arg-type]
    if not boxes:
        return "unknown", None

    encs = face_recognition.face_encodings(crop_rgb, known_face_locations=boxes)  # type: ignore[arg-type]
    if not encs:
        return "unknown", None

    face_enc = encs[0]  # first face in crop

    all_labels: List[str] = []
    all_encs: List[np.ndarray] = []
    for label, enc_list in known_encodings.items():
        for e in enc_list:
            all_labels.append(label)
            all_encs.append(e)

    if not all_encs:
        return "unknown", None

    distances = face_recognition.face_distance(all_encs, face_enc)  # type: ignore[arg-type]
    idx_best = int(np.argmin(distances))
    best_label = all_labels[idx_best]
    best_dist = float(distances[idx_best])

    if best_dist <= tolerance:
        return best_label, best_dist
    else:
        return "unknown", best_dist
