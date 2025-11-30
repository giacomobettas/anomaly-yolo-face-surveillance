import argparse
import os
from typing import Tuple

import tensorflow as tf

from src.ae_model import build_autoencoder
from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the autoencoder on NORMAL person crops for a given camera."
    )

    parser.add_argument(
        "--camera_id",
        type=str,
        required=True,
        help="Identifier for the camera (used only for logging).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with normal crops for this camera (e.g. data/cam1/normal_crops).",
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
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30).",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=128,
        help="Latent dimension of the bottleneck (default: 128).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/ae_best.weights.h5",
        help="Path to save best AE weights (default: checkpoints/ae_best.weights.h5).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/ae_person_autoencoder.keras",
        help="Path to save final AE model (default: models/ae_person_autoencoder.keras).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience on val_loss (default: 5).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of data used for validation (default: 0.2).",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Optional path to existing AE weights to resume training from.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    image_size: Tuple[int, int] = (args.image_size[0], args.image_size[1])
    channels = 3 if args.color_mode == "rgb" else 1
    input_shape = (image_size[0], image_size[1], channels)

    print(f"Camera ID: {args.camera_id}")
    print(f"Training AE on directory: {args.data_dir}")
    print(f"Image size: {image_size[0]}x{image_size[1]}, channels: {channels}")
    print(f"Saving best weights to: {args.checkpoint_path}")
    print(f"Saving final model to: {args.model_path}")

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    # Create training dataset from directory
    # We treat all images as a single class (no labels needed)
    dataset = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        labels=None,
        label_mode=None,
        image_size=image_size,
        color_mode=args.color_mode,
        batch_size=args.batch_size,
        shuffle=True,
        validation_split=args.val_split,
        subset="both",
        seed=42,
    )

    train_ds, val_ds = dataset

    # Normalize to [0,1] and map (x -> (x,x)) for AE
    def prep(x):
        x = tf.cast(x, tf.float32) / 255.0
        return x, x

    train_ds = train_ds.map(prep)
    val_ds = val_ds.map(prep)

    # Build model
    model = build_autoencoder(input_shape=input_shape, latent_dim=args.latent_dim)
    model.summary()

    # Optionally resume from existing weights
    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"Resuming AE training from weights: {args.resume_from}")
            model.load_weights(args.resume_from)
        else:
            print(f"[WARN] resume_from path does not exist: {args.resume_from}. Starting from scratch.")

    # Prepare dirs and callbacks
    ensure_dir(args.checkpoint_path)
    ensure_dir(args.model_path)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=max(1, args.patience // 2),
        verbose=1,
    )

    callbacks = [checkpoint_cb, early_stop_cb, reduce_lr_cb]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    model.save(args.model_path)
    print(f"Final AE model saved to: {args.model_path}")

    final_loss = history.history.get("loss", ["?"])[-1]
    final_val_loss = history.history.get("val_loss", ["?"])[-1]
    print(f"Final training loss: {final_loss}")
    print(f"Final validation loss: {final_val_loss}")


if __name__ == "__main__":
    main()
