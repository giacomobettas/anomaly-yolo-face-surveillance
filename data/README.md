# Data folder

This repository expects **user-provided data** (real datasets are not included).
To make the project runnable out of the box, a tiny **synthetic demo** crop set is included for quick smoke-testing.

## Expected layout

### Autoencoder training data (per camera)

Provide **normal person crops** (frames already cropped around a person) for each camera:

```text
data/
  cam1/
    normal_crops/
      demo/                # Optional tiny synthetic demo set
        *.jpg / *.png
      ...                  # Your own normal crops
```

Notes:

* Crops are resized to `--image_size` during training (default: `128x64`), so exact input size is not required.
* Training should contain **only normal behavior** for that camera/scene.

### Optional: face identity gallery

Face identification is optional and requires extra dependencies.

```text
data/
  faces/
    person1/
      *.jpg / *.png
    person2/
      *.jpg / *.png
```

If optional face dependencies are not installed, identities degrade gracefully to `"unknown"` and the rest of the pipeline still runs.

## Videos

Videos can be stored anywhere; you pass the path via CLI (you do not need to keep them under `data/`):

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
