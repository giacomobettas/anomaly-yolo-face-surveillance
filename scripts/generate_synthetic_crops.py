import argparse
import os
from typing import Tuple

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def generate_person_like_crop(
    size_hw: Tuple[int, int] = (128, 64),
    rgb: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a synthetic "person-crop-like" image:
    - dark background
    - bright ellipse (silhouette-ish)
    - small random shifts and mild noise
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    H, W = size_hw
    if rgb:
        img = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        img = np.zeros((H, W), dtype=np.uint8)

    # Random ellipse parameters (rough silhouette)
    cx = int(W * 0.5 + rng.integers(-6, 7))
    cy = int(H * 0.55 + rng.integers(-10, 11))
    ax = int(W * 0.18 + rng.integers(-2, 3))   # semi-axis x
    ay = int(H * 0.30 + rng.integers(-6, 7))   # semi-axis y

    color = (255, 255, 255) if rgb else 255
    cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, color, -1)

    # Optional "head" blob
    head_cx = int(cx + rng.integers(-3, 4))
    head_cy = int(cy - ay + rng.integers(-4, 5))
    head_r = max(4, int(min(W, H) * 0.06 + rng.integers(-1, 2)))
    cv2.circle(img, (head_cx, head_cy), head_r, color, -1)

    # Mild blur + noise
    if rng.random() < 0.9:
        img = cv2.GaussianBlur(img, (5, 5), 0)

    noise_level = float(rng.uniform(0.0, 12.0))
    if noise_level > 0:
        noise = rng.normal(0, noise_level, img.shape).astype(np.float32)
        img_f = img.astype(np.float32) + noise
        img = np.clip(img_f, 0, 255).astype(np.uint8)

    return img


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a tiny synthetic demo dataset of person-crop-like images."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/cam1/normal_crops/demo",
        help="Output directory (default: data/cam1/normal_crops/demo).",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=50,
        help="Number of images to generate (default: 50).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Image height (default: 128).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=64,
        help="Image width (default: 64).",
    )
    parser.add_argument(
        "--color_mode",
        type=str,
        choices=["rgb", "grayscale"],
        default="rgb",
        help='Color mode to generate (default: "rgb").',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    ensure_dir(args.out_dir)
    rgb = args.color_mode == "rgb"
    size_hw = (args.height, args.width)

    rng = np.random.default_rng(args.seed)

    for i in range(args.num_images):
        # vary seed per image but remain reproducible
        img = generate_person_like_crop(size_hw=size_hw, rgb=rgb, seed=int(rng.integers(0, 1_000_000)))
        out_path = os.path.join(args.out_dir, f"synthetic_{i:04d}.jpg")
        cv2.imwrite(out_path, img)

    print(f"[OK] Wrote {args.num_images} synthetic crops to: {args.out_dir}")


if __name__ == "__main__":
    main()
