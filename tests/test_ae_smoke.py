import numpy as np

from src.ae_model import build_autoencoder


def test_ae_build_and_forward_pass():
    """
    Smoke test:
    - Build the AE with default input shape.
    - Run a small random batch through it.
    """
    model = build_autoencoder()
    input_shape = model.input_shape[1:]  # (H, W, C)

    X = np.random.rand(2, *input_shape).astype("float32")
    Y = model.predict(X, verbose=0)

    assert Y.shape == X.shape
