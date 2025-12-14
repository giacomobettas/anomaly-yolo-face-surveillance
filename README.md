# Real-Time-Oriented CCTV Anomaly Scoring (YOLOv8 + Optional Face ID + Per-Camera Autoencoder + FFT Motion Feature)

End-to-end **video anomaly scoring** pipeline for home/elderly surveillance scenarios:

- **YOLOv8** person detection
- optional **face-based identity** (graceful fallback if unavailable)
- **per-camera convolutional autoencoder** trained on *normal* person crops
- **FFT motion feature** over the normalized vertical bbox center trajectory
- **interpretable anomaly scoring** + CSV logging + optional annotated video output

This repository represents the **final iteration** of my **thesis internship project** in video-based anomaly detection for elderly home surveillance, focused on **clean engineering**, **reproducibility**, and **realistic CCTV constraints**.

---

## Pipeline at a glance

For each frame in a video:

1. Detect people with YOLO
2. (Optional) identify a person via face recognition (else `"unknown"`)
3. Crop each detected person and feed the crop into a trained autoencoder
4. Compute reconstruction error (primary anomaly signal)
5. Compute supporting features:
   - posture score (soft bbox-geometry feature)
   - FFT motion score (default: 10s window) from normalized vertical bbox center
6. Combine into an anomaly score and flag
7. Write outputs (CSV log + optional annotated video)

---

## Key ideas and design choices

### Per-camera modeling
Autoencoders are trained **per camera** because each scene has its own “normal baseline”:
lighting, perspective, background, and typical movement patterns differ across cameras.  
This makes per-camera training a practical approach for surveillance deployments.

### Unsupervised training (normal-only)
The autoencoder is trained on **normal** person crops only. At inference time, “unseen” behaviors
(e.g., falls, climbing, unusual motion) tend to yield higher reconstruction error.

### Supporting features (not hard rules)
- **Posture** is used as a **weak supporting feature** derived from bbox geometry. It is intentionally *not*
  a hard fall detector (to reduce false positives such as lying on a bed).
- **FFT motion** summarizes short-window motion patterns (frequency-domain signature) using the
  normalized vertical bbox center over time.

### Optional identity
Face recognition is optional. If `face_recognition` is missing, the pipeline remains usable:
identities simply degrade to `"unknown"`.

---

## What this repo provides

### Training (per camera)
- AE training on normal person crops with checkpoints and final model export:
  - `src/ae_train.py`

### Inference / processing
- End-to-end video processing:
  - YOLO detections (`src/detector.py`)
  - optional face identity (`src/face_id.py`)
  - AE reconstruction error per person crop
  - FFT motion feature (`src/fft_buffer.py`)
  - combined anomaly scoring (`src/anomaly_rules.py`)
  - CSV logs + optional annotated video (`src/process_video.py`)
  - AE-only inference utilities for debugging/evaluation (`src/ae_infer.py`)


### Tests
- Lightweight smoke tests (AE forward pass, FFT buffer sanity, optional face test skip).

---

## Repository structure

```text
anomaly-yolo-face-surveillance/
  src/
    __init__.py
    ae_model.py            # Conv autoencoder architecture (encoder/decoder)
    ae_train.py            # Train AE on normal person crops (per-camera) + checkpoints
    detector.py            # YOLOv8 person detector wrapper (Ultralytics)
    face_id.py             # Optional face ID (graceful fallback if face_recognition missing)
    fft_buffer.py          # Per-identity rolling motion buffer + FFT-based motion score
    anomaly_rules.py       # Scoring logic: recon + posture + FFT -> combined score + anomaly flag
    process_video.py       # End-to-end pipeline: video -> detections -> scores -> CSV/video outputs
    ae_infer.py            # AE-only inference utilities (reconstruction error on crops/images)
    utils.py               # Utilities (path helpers, normalization, mse)
    video_io.py            # Video open/write helpers + frame annotation overlay

  tests/
    test_ae_smoke.py            # AE build + forward-pass smoke test
    test_face_id_smoke.py       # Face ID smoke test (skips if face_recognition not installed)
    test_fft_buffer_smoke.py    # FFTBuffer sanity test (scores in [0,1], no crashes)
    test_video_io_smoke.py      # Video writer/reader smoke test

  notebooks/
    colab_demo.ipynb       # Drive-persistent Colab demo (train + inference + outputs)
  
  scripts/
    generate_synthetic_crops.py  # Generates tiny synthetic crop demo dataset

  data/
    README.md              # Data layout + demo dataset notes
    faces/
      README.md            # Optional face gallery instructions

  requirements.txt                  # core dependencies (YOLO + AE + scoring)
  requirements-optional-face.txt    # optional Face ID deps (face_recognition + dlib)
  README.md
  LICENSE
```

---

## Data layout

This repo expects **pre-extracted person crops** for training (frames on disk).
Video-to-crops extraction can be handled upstream; this repo focuses on **training + scoring**.

### Autoencoder training (normal crops per camera)

```text
data/
  cam1/
    normal_crops/
      demo/                # Optional tiny synthetic demo set
        *.jpg / *.png
      ...                  # Your own normal crops
```

### Optional: known faces (for identity)

```text
data/
  faces/
    person1/
      *.jpg / *.png
    person2/
      *.jpg / *.png
```

### Videos (for end-to-end processing)

Videos can be stored anywhere; you provide the path via CLI:

```bash
python -m src.process_video --video_path /path/to/video.mp4 ...
```

## Synthetic demo crops

`data/cam1/normal_crops/demo/` contains a small synthetic set meant only for:

* verifying the training pipeline runs
* running smoke tests quickly

You can regenerate this demo set using the script:

```bash
python scripts/generate_synthetic_crops.py
```
Example:
```bash
python scripts/generate_synthetic_crops.py --num_images 50 --height 128 --width 64 --color_mode rgb
```

---

## Installation

Install core dependencies:

```bash
pip install -r requirements.txt
```

### Optional: face recognition notes

Face ID uses `face_recognition`, which depends on `dlib` and may require additional system tooling
on some platforms (especially Windows). The repo is designed so that:

* if `face_recognition` is missing, identity falls back to `"unknown"`;
* YOLO + AE + FFT scoring still run.

If you want identity support, install the optional face dependencies:

```bash
pip install -r requirements-optional-face.txt
```

---

## Quickstart (local)

### 1) Train the autoencoder (per camera)

```bash
python -m src.ae_train \
  --camera_id cam1 \
  --data_dir data/cam1/normal_crops \
  --image_size 128 64 \
  --color_mode rgb \
  --batch_size 16 \
  --epochs 30 \
  --checkpoint_path checkpoints/cam1_ae_best.weights.h5 \
  --model_path models/cam1_ae_person_autoencoder.keras
```

Tip: you can resume training with `--resume_from checkpoints/...`.

### 2) Process a video end-to-end (YOLO + AE + FFT scoring)

```bash
python -m src.process_video \
  --camera_id cam1 \
  --video_path videos/cam1_example.mp4 \
  --ae_model_path models/cam1_ae_person_autoencoder.keras \
  --output_video outputs/cam1_annotated.avi \
  --output_csv outputs/cam1_log.csv \
  --image_size 128 64 \
  --color_mode rgb \
  --fft_window_seconds 10.0
```

---

## Anomaly scoring

For each detected person:

1. **Reconstruction error (primary)**
   The AE reconstructs the person crop; we compute pixel-wise MSE between input and reconstruction.

2. **Posture score (supporting)**
   A soft feature derived from bbox geometry (normalized by frame height). This is intentionally not a hard
   fall rule, to reduce false positives (e.g., a person lying on a bed).

3. **FFT motion score (supporting)**
   We maintain a rolling window of the normalized vertical bbox center:

   ```math
   center\_y\_norm(t) = \frac{\frac{y_1 + y_2}{2}}{\text{frame\_height}}
   ```

   and compute an FFT-derived score over the last ~N seconds (default: 10s).

### Combined score

The final score is a weighted combination:

```math
score = w_{\text{recon}}\cdot recon\_score + w_{\text{posture}}\cdot posture\_score + w_{\text{fft}}\cdot fft\_score
```

All weights and thresholds are CLI-configurable:

* `--w_recon` (default `0.6`)
* `--w_posture` (default `0.25`)
* `--w_fft` (default `0.15`)
* `--recon_max` (default `0.1`) to map reconstruction error into `[0,1]`
* `--anomaly_threshold` (default `0.5`)

---

## Outputs

### CSV log

If `--output_csv` is provided, the pipeline writes per-detection rows including:

* bbox coordinates and confidence
* identity label (or `"unknown"`)
* reconstruction MSE
* recon/posture/fft sub-scores
* combined score
* anomaly flag

### Annotated video

If `--output_video` is provided, the pipeline writes an annotated video overlaying identity + confidence + combined score.

---

## Google Colab demo (Drive-persistent)

A Colab workflow is provided at:

```text
notebooks/colab_demo.ipynb
```

It mounts Google Drive and saves **models**, **checkpoints**, and **outputs** to Drive to survive Colab runtime resets.

Recommended Drive structure:

```text
MyDrive/anomaly_yolo_face_surveillance/
  data/
    cam1/normal_crops/*.jpg
    faces/ (optional)
  videos/
    cam1_example.mp4
  checkpoints/
  models/
  outputs/
```

In Colab, you can optionally install face recognition by running:
`pip install -r requirements-optional-face.txt`
(and installing any required system packages).

---

## Testing

```bash
python -m pytest tests/test_ae_smoke.py
python -m pytest tests/test_fft_buffer_smoke.py
```

Face ID smoke test is skipped automatically if `face_recognition` is not installed:

```bash
python -m pytest tests/test_face_id_smoke.py
```

---

## Notes on deployment scope

This repository focuses on:

* per-camera model training
* interpretable anomaly scoring
* logging/annotation for analysis and evaluation

A full production deployment typically adds:

* stable multi-person tracking IDs across frames
* multi-stream ingestion (RTSP/WebRTC) and orchestration
* alert delivery (webhook/SMS/email) with monitoring and audit logs
* calibration/validation pipelines for thresholds per camera

Those components are intentionally **out of scope** here to keep the repository clear and reproducible.

---

## License

Released under the license in the `LICENSE` file.