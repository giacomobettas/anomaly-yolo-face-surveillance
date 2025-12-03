"""
FFT-based motion feature for person trajectories.

We keep, per-identity, a ring buffer of the normalized vertical center
of the person bounding box (center_y / frame_height). When enough samples
are available, we compute an FFT-based score that summarizes how much
energy is in the low-frequency band.

This is meant to be a supporting anomaly feature, not a standalone
fall detector: unusual motion patterns will get different spectra than
ordinary walking or static posture.
"""

from collections import deque
from typing import Deque, Dict

import numpy as np


class FFTBuffer:
    """
    Maintain a rolling history of 1D motion signals per identity, and
    provide a normalized low-frequency FFT score in [0, 1].

    Typical usage:
        buf = FFTBuffer(max_len=250)
        buf.add("person1", center_y_norm)
        score = buf.get_fft_score("person1")
    """

    def __init__(self, max_len: int = 250):
        """
        Args:
            max_len: Maximum number of samples to retain per identity.
                     This should correspond roughly to fps * window_seconds.
        """
        if max_len < 8:
            max_len = 8  # need enough samples for FFT
        self.max_len = max_len
        self.buffers: Dict[str, Deque[float]] = {}

    def add(self, key: str, value: float) -> None:
        """
        Append a new sample for the given identity.

        Args:
            key: Identity key (e.g., face label or camera-local track id).
            value: Normalized motion value (e.g., center_y / frame_height).
        """
        dq = self.buffers.setdefault(key, deque(maxlen=self.max_len))
        dq.append(float(value))

    def get_fft_score(self, key: str) -> float:
        """
        Compute an FFT-based motion score for the given identity.

        Returns:
            A score in [0, 1], where higher means more low-frequency
            energy relative to total non-DC energy. If there are not
            enough samples or the signal is (near) constant, returns 0.0.
        """
        dq = self.buffers.get(key)
        if dq is None or len(dq) < 8:
            return 0.0

        x = np.asarray(dq, dtype=np.float32)
        x = x - x.mean()

        if np.allclose(x, 0.0):
            return 0.0

        fft = np.fft.rfft(x)
        mag = np.abs(fft)

        # mag[0] is DC; ignore it
        if mag.shape[0] <= 1:
            return 0.0

        non_dc = mag[1:]
        total = float(non_dc.sum())
        if total <= 1e-8:
            return 0.0

        # Take the first quarter of non-DC bins as "low frequency"
        k = max(1, non_dc.shape[0] // 4)
        low = float(non_dc[:k].sum())

        score = low / total  # already in [0, 1]
        # Numerical safety
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0

        return score
