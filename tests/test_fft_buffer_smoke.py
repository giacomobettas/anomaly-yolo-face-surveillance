import numpy as np

from src.fft_buffer import FFTBuffer


def test_fft_buffer_smoke():
    """
    Smoke test for FFTBuffer: ensure scores are in [0,1] and do not crash.
    """
    buf = FFTBuffer(max_len=32)

    key = "person1"
    # Simulate a simple motion pattern (increasing center_y_norm)
    for i in range(32):
        val = i / 31.0  # goes from 0 to 1
        buf.add(key, val)

    score = buf.get_fft_score(key)

    assert 0.0 <= score <= 1.0
