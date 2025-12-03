"""
Anomaly scoring rules: combine reconstruction error, posture, and FFT motion.

This module centralizes how we turn per-person features into:
- normalized sub-scores in [0, 1]
- a combined anomaly score
- a binary anomaly flag

The idea is:
- Reconstruction error is the primary driver.
- Posture (bbox geometry) is a weak supporting feature.
- FFT motion is an additional supporting feature.

The final combined score = w_recon * recon_score
                          + w_posture * posture_score
                          + w_fft * fft_score
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import numpy as np


@dataclass
class AnomalyConfig:
    """
    Configuration for anomaly scoring.

    w_recon: Weight for normalized reconstruction error.
    w_posture: Weight for posture-based score.
    w_fft: Weight for FFT-based motion score.
    recon_max: Value used to normalize reconstruction error:
               recon_score = min(recon_error / recon_max, 1.0)
               (this should be chosen based on validation).
    threshold: Combined score threshold for flagging anomaly.
    """

    w_recon: float = 0.6
    w_posture: float = 0.25
    w_fft: float = 0.15
    recon_max: float = 0.1
    threshold: float = 0.5


def normalize_reconstruction_error(
    recon_error: Optional[float],
    recon_max: float,
) -> float:
    """
    Normalize reconstruction error to [0, 1] using a simple linear scaling.

    If recon_error is None, returns 0.0.

    Note:
        recon_max should be set using validation on NORMAL data, e.g.
        a high quantile of reconstruction errors, so that normal frames
        map to scores mostly below 1.0.
    """
    if recon_error is None:
        return 0.0
    if recon_error <= 0.0:
        return 0.0
    if recon_max <= 0.0:
        return 1.0

    score = recon_error / recon_max
    if score > 1.0:
        score = 1.0
    return float(score)


def compute_posture_score(
    bbox: Tuple[float, float, float, float],
    frame_height: int,
) -> float:
    """
    Compute a soft posture score in [0, 1] based on bbox geometry.

    We do NOT treat posture as a hard fall detector. Instead:
    - We compute the relative bbox height and bottom position.
    - We consider deviations from a "typical standing" range as more unusual.
    - The score is 0 for clearly normal ranges and increases smoothly.

    Args:
        bbox: (x1, y1, x2, y2)
        frame_height: height of the original frame in pixels.

    Returns:
        posture_score in [0, 1]
    """
    x1, y1, x2, y2 = bbox
    h = max(1.0, float(frame_height))
    bbox_height = max(0.0, float(y2) - float(y1))
    height_ratio = bbox_height / h
    bottom_ratio = float(y2) / h

    # Typical ranges for a standing person in a fixed camera:
    # These are heuristic and meant to be soft.
    # - height_ratio around 0.4 is "normal"
    # - bottom_ratio somewhere mid-frame to 0.8 is "normal"
    height_center = 0.4
    height_tol = 0.2  # +/- 0.2 around center considered mostly normal

    bottom_center = 0.75
    bottom_tol = 0.25

    def deviation_score(value: float, center: float, tol: float) -> float:
        """
        Compute |value-center| / tol, clipped to [0,1].
        """
        diff = abs(value - center)
        if diff <= tol:
            return 0.0
        score = (diff - tol) / tol  # 0 at tol, 1 at 2*tol or beyond
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0
        return float(score)

    height_score = deviation_score(height_ratio, height_center, height_tol)
    bottom_score = deviation_score(bottom_ratio, bottom_center, bottom_tol)

    # Take the maximum: if either height or bottom is clearly unusual, posture is unusual.
    posture_score = max(height_score, bottom_score)
    return posture_score


def compute_anomaly_scores(
    bbox: Tuple[float, float, float, float],
    frame_shape: Tuple[int, int, int],
    recon_error: Optional[float],
    fft_score: float,
    config: AnomalyConfig,
) -> Dict[str, Any]:
    """
    Compute all anomaly-related scores for a single detection.

    Args:
        bbox: Person bounding box (x1, y1, x2, y2) in original frame coordinates.
        frame_shape: (H, W, C) of the original frame.
        recon_error: Reconstruction MSE for the crop (or None).
        fft_score: FFT-based motion score in [0,1].
        config: AnomalyConfig with weights and threshold.

    Returns:
        Dict with keys:
            - recon_score
            - posture_score
            - fft_score
            - combined_score
            - is_anomaly (bool)
    """
    frame_height = frame_shape[0]

    recon_score = normalize_reconstruction_error(recon_error, config.recon_max)
    posture_score = compute_posture_score(bbox, frame_height)

    # Clamp fft_score to [0,1] just in case
    if fft_score < 0.0:
        fft_score_clamped = 0.0
    elif fft_score > 1.0:
        fft_score_clamped = 1.0
    else:
        fft_score_clamped = float(fft_score)

    combined = (
        config.w_recon * recon_score
        + config.w_posture * posture_score
        + config.w_fft * fft_score_clamped
    )

    is_anomaly = combined >= config.threshold

    return {
        "recon_score": recon_score,
        "posture_score": posture_score,
        "fft_score": fft_score_clamped,
        "combined_score": float(combined),
        "is_anomaly": bool(is_anomaly),
    }
