import os
import tempfile

import cv2
import numpy as np

from src.video_io import create_video_writer, open_video_capture


def test_video_io_smoke():
    """
    Smoke test for video writer/reader.

    - Create a temporary video with a few black frames.
    - Reopen it and read at least one frame.
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        video_path = os.path.join(tmp_dir, "test_video.avi")

        fps = 10.0
        frame_size = (320, 240)
        writer = create_video_writer(video_path, fps, frame_size)

        for _ in range(5):
            frame = np.zeros((frame_size[1], frame_size[0], 3), dtype="uint8")
            writer.write(frame)
        writer.release()

        cap, fps_read, size_read = open_video_capture(video_path)
        ret, frame = cap.read()
        cap.release()

        assert ret is True
        assert frame is not None
    finally:
        # cleanup handled by OS; no strict need to remove files here
        pass
