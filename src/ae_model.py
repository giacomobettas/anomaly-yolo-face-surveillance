"""
Autoencoder model for person-crop reconstruction.

This AE is trained per-camera on NORMAL crops extracted via YOLO
so that at inference time reconstruction error can indicate anomalies
(unseen motion patterns, falls, strange poses, etc.).
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_autoencoder(
    input_shape: Tuple[int, int, int] = (128, 64, 3),
    latent_dim: int = 128
) -> tf.keras.Model:
    """
    Build a convolutional autoencoder for person crops.

    Args:
        input_shape: (H, W, C) input image shape.
        latent_dim: Size of the latent vector. Default: 128.

    Returns:
        A compiled Keras Model.
    """

    inputs = layers.Input(shape=input_shape, name="input_crop")

    # ---- Encoder ----
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)

    # Bottleneck
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    # ---- Decoder ----
    x = layers.Dense((input_shape[0]//8) * (input_shape[1]//8) * 128, activation="relu")(latent)
    x = layers.Reshape((input_shape[0]//8, input_shape[1]//8, 128))(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)

    outputs = layers.Conv2D(
        input_shape[2],
        3,
        activation="sigmoid",
        padding="same",
        name="reconstruction"
    )(x)

    model = models.Model(inputs, outputs, name="person_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

    return model


if __name__ == "__main__":
    # Smoke test
    ae = build_autoencoder()
    ae.summary()
