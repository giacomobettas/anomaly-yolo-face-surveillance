import argparse
import os
from typing import Tuple, List

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.utils import normalize_img, mse, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute reconstruction errors for person crops using a trained AE."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with crops to evaluate (e.g. data/cam1/eval_crops).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained AE model (.keras).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[128, 64],
        help="Image size as HEIGHT WIDTH (default: 128 64).",
    )
    parser.add_argument(
        "--color_mode",
        type=str,
        choices=["rgb", "grayscale"],
        default="rgb",
        help='Color mode for loading images (default: "rgb").',
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Optional path to CSV file for per-image reconstruction errors.",
    )
    parser.add_argument(
        "--show_hist",
        action="store_true",
        help="If set, show a histogram of reconstruction errors.",
    )

    return parser.parse_args()


def load_images_from_dir(
    root_dir: str,
    image_size: Tuple[int, int],
    color_mode: str,
) -> (np.ndarray, List[str]):
    """
    Load all jpg/png images from directory (non-recursive), resize, normalize.
    Returns: (images, filenames)
    """
    paths: List[str] = []
    for fname in sorted(os.listdir(root_dir)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            paths.append(os.path.join(root_dir, fname))

    if not paths:
        raise RuntimeError(f"No images found in {root_dir}")

    imgs = []
    filenames = []
    for p in paths:
        if color_mode == "grayscale":
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(p, cv2.IMREAD_COLOR)

        if img is None:
            continue

        img = cv2.resize(img, (image_size[1], image_size[0]))  # cv2: (W,H)
        if color_mode == "grayscale":
            img = np.expand_dims(img, axis=-1)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = normalize_img(img)
        imgs.append(img)
        filenames.append(os.path.basename(p))

    if not imgs:
        raise RuntimeError(f"No valid images could be loaded from {root_dir}")

    X = np.stack(imgs, axis=0)
    return X, filenames


def main():
    args = parse_args()

    image_size: Tuple[int, int] = (args.image_size[0], args.image_size[1])
    channels = 3 if args.color_mode == "rgb" else 1

    print(f"Loading AE model from: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path, compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

    print(f"Loading images from: {args.data_dir}")
    X, filenames = load_images_from_dir(
        args.data_dir,
        image_size=image_size,
        color_mode=args.color_mode,
    )

    if X.shape[-1] != channels:
        raise ValueError(f"Channel mismatch: data has {X.shape[-1]} channels, but color_mode implies {channels}.")

    print(f"Evaluating {X.shape[0]} images...")
    preds = model.predict(X, verbose=1)

    errors = []
    for i in range(X.shape[0]):
        errors.append(mse(X[i], preds[i]))

    errors = np.array(errors, dtype="float32")

    print("\nReconstruction error statistics:")
    print(f"  N images: {errors.shape[0]}")
    print(f"  Mean:     {errors.mean():.6f}")
    print(f"  Std:      {errors.std():.6f}")
    print(f"  Min:      {errors.min():.6f}")
    print(f"  Max:      {errors.max():.6f}")

    # Optional histogram
    if args.show_hist:
        plt.figure(figsize=(6, 4))
        plt.hist(errors, bins=30, alpha=0.8)
        plt.xlabel("Reconstruction MSE")
        plt.ylabel("Count")
        plt.title("Reconstruction error distribution")
        plt.tight_layout()
        plt.show()

    # Optional CSV
    if args.output_csv:
        import csv

        print(f"\nSaving per-image errors to CSV: {args.output_csv}")
        ensure_dir(args.output_csv)

        with open(args.output_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "reconstruction_mse"])
            for fname, err in zip(filenames, errors):
                writer.writerow([fname, float(err)])

        print("CSV export completed.")


if __name__ == "__main__":
    main()
