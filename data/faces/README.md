# Faces folder (optional)

This folder is used only if you want **face-based identity recognition**.

Face ID is optional and requires installing:

```bash
pip install -r requirements-optional-face.txt
````

If the optional dependencies are not installed, the pipeline still runs and identities will degrade to `"unknown"`.

## Expected layout

```text
faces/
  person1/
    img1.jpg
    img2.jpg
  person2/
    img1.jpg
```

Guidelines:

* Each subfolder name is treated as the identity label.
* Use multiple images per person (recommended: **5–20**), varying lighting/angle.
* Images without detectable faces are skipped.
