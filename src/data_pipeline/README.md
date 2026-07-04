# Task 1 → Task 2 Integration Guide

**From:** Emmanuel Oringe (Task 1 – Data Preprocessing)
**To:** Peter Tobiko (Task 2 – Text Detection)

This document explains how to use the data pipeline outputs from Task 1
in your text detection work.

---

## Quick Start

```python
from src.data_pipeline.prep import get_dataloaders, split_dataset, ReceiptDataset

# Get ready-to-use PyTorch DataLoaders (train has augmentation, val/test do not)
train_loader, val_loader, test_loader = get_dataloaders(
    data_dir="dataset",
    batch_size=8,
    target_size=(640, 640),   # Change if your detection model needs different dims
    mode="detection",         # Keeps RGB — best for detection models
)
```

---

## What Each Batch Contains

When you iterate a DataLoader, each batch is a **dictionary** with these keys:

| Key | Type | Shape / Description |
|-----|------|---------------------|
| `images` | `torch.Tensor` | `(B, 3, H, W)` float32 in `[0, 1]` — preprocessed receipt images |
| `original_images` | `list[PIL.Image]` | Original unmodified images (for visualization) |
| `boxes` | `list[np.ndarray]` | Each element is `(N, 8)` int32 — **original** coordinate bounding boxes |
| `scaled_boxes` | `list[np.ndarray]` | Each element is `(N, 8)` int32 — boxes transformed to match `images` |
| `transcripts` | `list[list[str]]` | Text transcript for each bounding box |
| `keys` | `list[dict]` | Ground truth: `{"company", "date", "address", "total"}` |
| `filenames` | `list[str]` | Image filenames (e.g., `"000.jpg"`) |
| `scales` | `list[float]` | Scale factor applied during preprocessing |
| `pad_xs` | `list[int]` | Horizontal padding offset |
| `pad_ys` | `list[int]` | Vertical padding offset |

### Bounding Box Format

Each box is an array of 8 integers representing 4 corner points:

```
[x1, y1, x2, y2, x3, y3, x4, y4]
```

Where the points are ordered: **top-left → top-right → bottom-right → bottom-left**.

**Important:** Use `scaled_boxes` (not `boxes`) when training your detection model,
because `scaled_boxes` match the coordinate space of the preprocessed `images` tensor.
Use `boxes` only if you are working with `original_images`.

---

## Preprocessing Modes

The pipeline supports 3 preprocessing modes. Choose based on your detection model:

| Mode | Output | Best For |
|------|--------|----------|
| `"detection"` (default) | RGB image | CNN-based detectors (DBNet, EAST, YOLOv8) |
| `"recognition"` | 3-channel grayscale | OCR text recognition models |
| `"binary"` | Adaptive-threshold binary | Classical contour-based detection |

```python
# Example: binary mode for OpenCV contour detection
train_loader, _, _ = get_dataloaders("dataset", mode="binary")
```

---

## Typical Training Loop

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        images = batch["images"]          # (B, 3, 640, 640) tensor
        target_boxes = batch["scaled_boxes"]  # list of (N, 8) arrays

        # Your detection model forward pass
        predictions = model(images)

        # Compute loss against target_boxes
        loss = detection_loss(predictions, target_boxes)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## Using Individual Functions

If you need finer control instead of full DataLoaders:

### Load and preprocess a single image

```python
from src.data_pipeline.prep import load_image, preprocess_image, load_boxes, scale_boxes

# Load
img = load_image("dataset/img/000.jpg")

# Preprocess (resize + pad, aspect ratio preserved)
result = preprocess_image(img, target_size=(640, 640), mode="detection")
processed_img = result["image"]   # PIL Image, (640, 640)
scale = result["scale"]
pad_x = result["pad_x"]
pad_y = result["pad_y"]

# Load and transform boxes to match processed image
boxes, transcripts = load_boxes("dataset/box/000.csv")
scaled = scale_boxes(boxes, scale, pad_x, pad_y)
```

### Convert preprocessed image to numpy/tensor

```python
import numpy as np
import torch

arr = np.array(processed_img, dtype=np.float32) / 255.0  # (H, W, 3)
tensor = torch.from_numpy(arr).permute(2, 0, 1)          # (3, H, W)
```

### Get the train/val/test split file lists

```python
from src.data_pipeline.prep import split_dataset

train_files, val_files, test_files = split_dataset("dataset")
# train_files = ["152.jpg", "403.jpg", ...]  (80% of 626 = 500 files)
# val_files   = [...]                         (10% = 62 files)
# test_files  = [...]                         (10% = 64 files)
```

The split is **deterministic** (seed=42) so all team members get the same splits.

---

## Converting Boxes Back to Original Coordinates

If you need to map detection predictions back to original image space
(e.g., for visualization or passing results to Task 3):

```python
def unscale_boxes(scaled_boxes, scale, pad_x, pad_y):
    """Inverse of scale_boxes — maps preprocessed coords back to original."""
    result = scaled_boxes.astype(np.float64).copy()
    for i in range(4):
        result[:, i * 2]     = (result[:, i * 2] - pad_x) / scale
        result[:, i * 2 + 1] = (result[:, i * 2 + 1] - pad_y) / scale
    return result.astype(np.int32)
```

---

## File Layout Reference

```
dataset/
├── img/          # 626 receipt JPEGs (variable sizes, e.g. 463×1013 to 4961×7016)
├── box/          # 626 CSV files: x1,y1,x2,y2,x3,y3,x4,y4,transcript
└── key/          # 626 JSON files: {"company", "date", "address", "total"}
```

---

## Questions?

Reach out to Emmanuel Oringe or check the unit tests at
`tests/test_data_pipeline.py` for working examples of every function.
