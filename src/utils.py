import os
import numpy as np


def ensure_dir(path: str):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def normalize_img(img: np.ndarray) -> np.ndarray:
    """Normalize image to float32 [0,1]."""
    return img.astype("float32") / 255.0


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute mean squared error between two images."""
    return float(np.mean((a - b) ** 2))
