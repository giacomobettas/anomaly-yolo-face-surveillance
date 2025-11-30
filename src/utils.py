import os
import numpy as np


def ensure_dir(path: str):
    """
    Ensure that the directory for the given path exists.

    - If 'path' looks like a file path (has an extension), create its parent directory.
    - If 'path' looks like a directory (no extension), create that directory.
    """
    # If path has an extension, assume it's a file and get its directory
    directory = path
    _, ext = os.path.splitext(path)
    if ext:  # has an extension -> treat as file path
        directory = os.path.dirname(path)

    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def normalize_img(img: np.ndarray) -> np.ndarray:
    """Normalize image to float32 [0,1]."""
    return img.astype("float32") / 255.0


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute mean squared error between two images."""
    return float(np.mean((a - b) ** 2))
