import argparse
import os
from typing import Tuple, List, Optional

import cv2
import numpy as np
import tensorflow as tf

from src.detector import YOLODetector
from src.face_id import load_face_encodings, identify_face, _FACE_LIB_AVAILABLE
from src.utils import ensure_dir, normalize_img, mse
from src.video_io import open_video_capture, create_video_writer, draw_detections
from src.fft_buffer import FFTBuffer
from src.anomaly_rules import AnomalyConfig, compute_anomaly_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process a video: YOLO person detection + (optional) face ID + "
            "autoencoder reconstruction error on person crops + anomaly scoring."
        )
    )

    parser.add_argument(
        "--camera_id",
        type=str,
        required=True,
        help="Camera identifier (used only for logging and filenames).",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to input video.",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default="",
        help="Optional path to save annotated output video.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Optional path to CSV log with per-frame detections and scores.",
    )
    parser.add_argument(
        "--ae_model_path",
        type=str,
        required=True,
        help="Path to trained AE model (.keras) for this camera.",
    )
    parser.add_argument(
        "--faces_root",
        type=str,
        default="",
        help=(
            "Root dir for known faces (e.g. data/faces). If empty or face_recognition "
            "is unavailable, identity will always be 'unknown'."
        ),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[128, 64],
        help="AE input size as HEIGHT WIDTH (default: 128 64).",
    )
    parser.add_argument(
        "--color_mode",
        type=str,
        choices=["rgb", "grayscale"],
        default="rgb",
        help='Color mode for AE crops (default: "rgb").',
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path or name (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=0.25,
        help="YOLO confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--iou_thres",
        type=float,
        default=0.45,
        help="YOLO IoU threshold (default: 0.45).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device hint for YOLO (e.g. 'cpu' or 'cuda').",
    )
    # Anomaly scoring config
    parser.add_argument(
        "--w_recon",
        type=float,
        default=0.6,
        help="Weight for normalized reconstruction error (default: 0.6).",
    )
    parser.add_argument(
        "--w_posture",
        type=float,
        default=0.25,
        help="Weight for posture-based score (default: 0.25).",
    )
    parser.add_argument(
        "--w_fft",
        type=float,
        default=0.15,
        help="Weight for FFT-based motion score (default: 0.15).",
    )
    parser.add_argument(
        "--recon_max",
        type=float,
        default=0.1,
        help=(
            "Reconstruction error value mapped to recon_score=1.0. "
            "Should be tuned on validation data (default: 0.1)."
        ),
    )
    parser.add_argument(
        "--anomaly_threshold",
        type=float,
        default=0.5,
        help="Combined score threshold for flagging anomaly (default: 0.5).",
    )
    parser.add_argument(
        "--fft_window_seconds",
        type=float,
        default=10.0,
        help="Time window in seconds for FFT motion feature (default: 10.0).",
    )

    return parser.parse_args()


def load_ae_model(model_path: str) -> tf.keras.Model:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"AE model not found: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def crop_person_for_ae(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
    color_mode: str,
) -> Optional[np.ndarray]:
    """
    Crop person region from frame and resize/normalize for AE.

    Returns:
        A numpy array of shape (H, W, C) in [0,1], or None if crop invalid.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Resize to (W, H) for cv2, then reshape to (H, W, C)
    H, W = image_size
    crop_resized = cv2.resize(crop, (W, H))  # cv2 expects (width, height)

    if color_mode == "grayscale":
        crop_gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
        crop_gray = np.expand_dims(crop_gray, axis=-1)
        crop_norm = normalize_img(crop_gray)
    else:
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        crop_norm = normalize_img(crop_rgb)

    return crop_norm


def main():
    args = parse_args()

    image_size: Tuple[int, int] = (args.image_size[0], args.image_size[1])
    channels = 3 if args.color_mode == "rgb" else 1

    print(f"[INFO] Camera ID: {args.camera_id}")
    print(f"[INFO] Input video: {args.video_path}")
    print(f"[INFO] AE model: {args.ae_model_path}")
    if args.output_video:
        print(f"[INFO] Annotated video will be saved to: {args.output_video}")
    if args.output_csv:
        print(f"[INFO] CSV log will be saved to: {args.output_csv}")

    # Load AE model
    ae_model = load_ae_model(args.ae_model_path)
    expected_shape = ae_model.input_shape[1:]
    if expected_shape != (image_size[0], image_size[1], channels):
        print(
            f"[WARN] AE model input shape {expected_shape} does not match "
            f"--image_size {image_size} and color_mode {args.color_mode}"
        )

    # Anomaly config
    anomaly_cfg = AnomalyConfig(
        w_recon=args.w_recon,
        w_posture=args.w_posture,
        w_fft=args.w_fft,
        recon_max=args.recon_max,
        threshold=args.anomaly_threshold,
    )
    print(
        f"[INFO] Anomaly config: w_recon={anomaly_cfg.w_recon}, "
        f"w_posture={anomaly_cfg.w_posture}, w_fft={anomaly_cfg.w_fft}, "
        f"recon_max={anomaly_cfg.recon_max}, threshold={anomaly_cfg.threshold}"
    )

    # Load face encodings (if library and path available)
    known_encodings = {}
    if args.faces_root and _FACE_LIB_AVAILABLE:
        try:
            print(f"[INFO] Loading face encodings from: {args.faces_root}")
            known_encodings = load_face_encodings(args.faces_root)
            print(f"[INFO] Loaded encodings for labels: {list(known_encodings.keys())}")
        except Exception as e:
            print(f"[WARN] Could not load face encodings: {e}")
            known_encodings = {}
    elif args.faces_root and not _FACE_LIB_AVAILABLE:
        print("[WARN] face_recognition not available; faces_root will be ignored.")
    else:
        print("[INFO] faces_root not provided; identities will be 'unknown'.")

    # Create YOLO detector
    detector = YOLODetector(
        model_path=args.yolo_model,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device=args.device,
    )

    # Open video
    cap, fps, frame_size = open_video_capture(args.video_path)
    frame_height, frame_width = frame_size[1], frame_size[0]

    # FFT buffer: size ≈ fps * window_seconds
    fft_window_frames = max(8, int(fps * args.fft_window_seconds))
    fft_buffer = FFTBuffer(max_len=fft_window_frames)
    print(
        f"[INFO] FFT window: {args.fft_window_seconds} s "
        f"({fft_window_frames} frames at {fps:.2f} fps)"
    )

    writer = None
    if args.output_video:
        ensure_dir(args.output_video)
        writer = create_video_writer(args.output_video, fps, frame_size)

    csv_file = None
    csv_writer = None
    if args.output_csv:
        import csv

        ensure_dir(args.output_csv)
        csv_file = open(args.output_csv, mode="w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "frame_idx",
                "time_sec",
                "x1",
                "y1",
                "x2",
                "y2",
                "conf",
                "identity",
                "reconstruction_mse",
                "recon_score",
                "posture_score",
                "fft_score",
                "combined_score",
                "is_anomaly",
            ]
        )

    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_h, frame_w = frame.shape[:2]

            # YOLO detections
            detections = detector.detect_persons(frame)

            identities: List[str] = []
            errors: List[Optional[float]] = []

            # Extended scores for annotation
            recon_scores: List[float] = []
            posture_scores: List[float] = []
            fft_scores: List[float] = []
            combined_scores: List[float] = []
            anomaly_flags: List[bool] = []

            for det in detections:
                bbox = det["bbox"]
                x1, y1, x2, y2 = bbox

                # AE crop and reconstruction error
                crop_norm = crop_person_for_ae(frame, bbox, image_size, args.color_mode)
                if crop_norm is not None:
                    X = np.expand_dims(crop_norm, axis=0)  # (1, H, W, C)
                    recon = ae_model.predict(X, verbose=0)
                    err = mse(X[0], recon[0])
                else:
                    err = None

                # Identity (face-based, if possible)
                identity, _ = identify_face(frame, bbox, known_encodings)

                # FFT motion feature: use normalized vertical center of bbox
                center_y = 0.5 * (y1 + y2)
                center_y_norm = float(center_y) / max(1.0, float(frame_h))
                fft_buffer.add(identity, center_y_norm)
                fft_score = fft_buffer.get_fft_score(identity)

                # Anomaly scores
                scores = compute_anomaly_scores(
                    bbox=bbox,
                    frame_shape=frame.shape,
                    recon_error=err,
                    fft_score=fft_score,
                    config=anomaly_cfg,
                )

                identities.append(identity)
                errors.append(err)
                recon_scores.append(scores["recon_score"])
                posture_scores.append(scores["posture_score"])
                fft_scores.append(scores["fft_score"])
                combined_scores.append(scores["combined_score"])
                anomaly_flags.append(scores["is_anomaly"])

                # CSV logging
                if csv_writer is not None:
                    time_sec = frame_idx / fps
                    csv_writer.writerow(
                        [
                            frame_idx,
                            float(time_sec),
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                            float(det.get("conf", 0.0)),
                            identity,
                            float(err) if err is not None else "",
                            scores["recon_score"],
                            scores["posture_score"],
                            scores["fft_score"],
                            scores["combined_score"],
                            int(scores["is_anomaly"]),
                        ]
                    )

            # Draw annotations (we at least display combined score and recon error)
            if writer is not None:
                # For drawing, we piggyback on 'errors' but we could also overlay combined_score.
                # Here, we override 'errors' with combined_scores to visualize anomaly strength.
                annotated = draw_detections(
                    frame,
                    detections,
                    identities=identities,
                    errors=combined_scores,
                )
                writer.write(annotated)

            frame_idx += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()

    print(f"[INFO] Finished processing {frame_idx} frames.")


if __name__ == "__main__":
    main()
