import os
import shutil
import tempfile
import numpy as np
import pytest

face_recognition = pytest.importorskip(
    "face_recognition", reason="face_recognition not installed; skipping face ID smoke test."
)

from src.face_id import load_face_encodings, identify_face


def test_face_id_smoke():
    """
    Smoke test for face ID utilities, without requiring real faces.
    """

    tmp_dir = tempfile.mkdtemp()
    faces_root = os.path.join(tmp_dir, "faces")
    os.makedirs(faces_root, exist_ok=True)

    try:
        encodings = load_face_encodings(faces_root)
        assert isinstance(encodings, dict)

        # Dummy frame and bbox
        frame = np.zeros((480, 640, 3), dtype="uint8")
        bbox = (100.0, 100.0, 200.0, 200.0)

        label, dist = identify_face(frame, bbox, encodings)
        assert label == "unknown"
    finally:
        shutil.rmtree(tmp_dir)
