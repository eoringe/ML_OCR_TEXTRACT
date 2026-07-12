"""
Task 1: Data Preprocessing & Loader
Author: Emmanuel Oringe
Branch: task-1-data-pipeline-emmanuel-oringe

This module provides the complete data preprocessing pipeline for the SROIE
receipt OCR project. It handles:
  - Loading receipt images and their annotations (bounding boxes, transcripts, keys)
  - Image preprocessing (resize with aspect ratio, grayscale, binarization, deskew)
  - Data augmentation (rotation, brightness/contrast jitter, noise injection)
  - PyTorch Dataset & DataLoader creation with reproducible train/val/test splits
  - A custom collate function for variable-length bounding box annotations

Usage for downstream tasks (e.g., Task 2 - Text Detection):
    from src.data_pipeline.prep import get_dataloaders, split_dataset, ReceiptDataset
    train_loader, val_loader, test_loader = get_dataloaders("dataset", batch_size=4)
"""

import os
import json
import random
import logging

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Optional imports — gracefully degrade if not installed
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TARGET_SIZE = (640, 640)  # (width, height)
VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
SROIE_FIELDS = ("company", "date", "address", "total")
RANDOM_SEED = 42


# ===========================================================================
#  Annotation Loaders
# ===========================================================================

def load_boxes(filepath):
    """
    Loads SROIE bounding-box annotation file (.csv or .txt).

    Each line has the format:
        x1,y1,x2,y2,x3,y3,x4,y4,transcription

    Args:
        filepath (str): Path to the annotation file.

    Returns:
        tuple: (boxes, transcripts)
            - boxes: np.ndarray of shape (N, 8) with int32 coordinates
            - transcripts: list[str] of N transcription strings
    """
    boxes = []
    transcripts = []

    if not os.path.exists(filepath):
        return np.zeros((0, 8), dtype=np.int32), []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on comma with a maximum of 8 splits so the transcript
            # (which may contain commas) remains intact.
            parts = line.split(",", 8)
            if len(parts) < 9:
                continue
            try:
                coords = [int(float(x)) for x in parts[:8]]
                transcript = parts[8].strip()
                boxes.append(coords)
                transcripts.append(transcript)
            except (ValueError, IndexError):
                logger.warning("Skipping malformed line in %s: %s", filepath, line)
                continue

    if boxes:
        return np.array(boxes, dtype=np.int32), transcripts
    return np.zeros((0, 8), dtype=np.int32), []


def load_keys(filepath):
    """
    Loads the SROIE key-information ground truth JSON.

    Expected JSON keys: company, date, address, total.

    Args:
        filepath (str): Path to the .json label file.

    Returns:
        dict: {"company": str, "date": str, "address": str, "total": str}
    """
    empty = {k: "" for k in SROIE_FIELDS}
    if not os.path.exists(filepath):
        return empty
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        # Ensure all expected keys exist
        return {k: str(data.get(k, "")).strip() for k in SROIE_FIELDS}
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Failed to load key file %s: %s", filepath, exc)
        return empty


# ===========================================================================
#  Image Preprocessing Functions
# ===========================================================================

def load_image(path):
    """
    Loads an image from disk as an RGB PIL Image.

    Args:
        path (str): Absolute or relative path to the image file.

    Returns:
        PIL.Image.Image: The loaded RGB image.

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If the file cannot be read as an image.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as exc:
        raise ValueError(f"Cannot read image {path}: {exc}") from exc


def resize_with_aspect_ratio(image, target_size=DEFAULT_TARGET_SIZE, fill_color=(0, 0, 0)):
    """
    Resizes an image to fit within *target_size* while maintaining aspect ratio.
    The remaining space is padded with *fill_color*.

    Args:
        image (PIL.Image.Image): Input RGB image.
        target_size (tuple): (width, height) of the output canvas.
        fill_color (tuple): RGB fill for the padding region.

    Returns:
        tuple: (resized_image, scale_factor, pad_x, pad_y)
            - resized_image: PIL.Image of exactly target_size
            - scale_factor: float by which the image was scaled
            - pad_x: int horizontal padding added on the left
            - pad_y: int vertical padding added on the top
    """
    target_w, target_h = target_size
    orig_w, orig_h = image.size

    # Compute scale factor to fit inside target while keeping ratio
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = image.resize((new_w, new_h), Image.LANCZOS)

    # Create canvas and paste centred
    canvas = Image.new("RGB", (target_w, target_h), fill_color)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    return canvas, scale, pad_x, pad_y


def to_grayscale(image):
    """
    Converts a PIL RGB image to grayscale and returns a 3-channel copy
    (compatible with CNN backbones that expect 3 channels).

    Args:
        image (PIL.Image.Image): RGB image.

    Returns:
        PIL.Image.Image: 3-channel grayscale image.
    """
    gray = ImageOps.grayscale(image)
    return Image.merge("RGB", (gray, gray, gray))


def binarize(image, threshold=128):
    """
    Applies a simple threshold binarization on a grayscale image.

    Args:
        image (PIL.Image.Image): Input RGB image.
        threshold (int): Pixel intensity threshold (0-255).

    Returns:
        PIL.Image.Image: Binarized 3-channel image.
    """
    gray = ImageOps.grayscale(image)
    binary = gray.point(lambda px: 255 if px > threshold else 0, "L")
    return Image.merge("RGB", (binary, binary, binary))


def adaptive_binarize(image, block_size=11, offset=2):
    """
    Applies adaptive thresholding using OpenCV (if available) for better
    receipt text separation. Falls back to simple binarization otherwise.

    Args:
        image (PIL.Image.Image): Input RGB image.
        block_size (int): Size of the pixel neighbourhood for threshold calculation.
        offset (int): Constant subtracted from the computed threshold.

    Returns:
        PIL.Image.Image: Binarized 3-channel image.
    """
    if not HAS_CV2:
        return binarize(image)

    arr = np.array(ImageOps.grayscale(image))
    binary = cv2.adaptiveThreshold(
        arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, offset
    )
    pil_bin = Image.fromarray(binary, "L")
    return Image.merge("RGB", (pil_bin, pil_bin, pil_bin))


def deskew(image, max_angle=10.0):
    """
    Attempts to correct slight image rotation (skew) caused by scanning.

    Uses OpenCV if available; otherwise returns the image unchanged.

    Args:
        image (PIL.Image.Image): Input RGB image.
        max_angle (float): Maximum absolute correction angle (degrees).

    Returns:
        PIL.Image.Image: De-skewed image.
    """
    if not HAS_CV2:
        return image

    arr = np.array(ImageOps.grayscale(image))
    # Use Hough line detection to estimate the dominant angle
    edges = cv2.Canny(arr, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return image

    angles = []
    for line in lines:
        if hasattr(line, "ndim") and line.ndim == 1:
            x1, y1, x2, y2 = line
        elif len(line) == 4 and not hasattr(line[0], "__len__"):
            x1, y1, x2, y2 = line
        else:
            x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < max_angle:
            angles.append(angle)

    if not angles:
        return image

    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:
        return image

    return image.rotate(median_angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))


def preprocess_image(image, target_size=DEFAULT_TARGET_SIZE, mode="detection"):
    """
    Full preprocessing pipeline for a single receipt image.

    Modes:
        - "detection":  Resize + pad → keep RGB (for text detection models).
        - "recognition": Resize + pad → grayscale 3-channel (for OCR models).
        - "binary":     Resize + pad → adaptive binarization (for classical OCR).

    Args:
        image (PIL.Image.Image): Input RGB image.
        target_size (tuple): (width, height) for the output.
        mode (str): One of "detection", "recognition", "binary".

    Returns:
        dict:
            - "image": PIL.Image.Image of size target_size
            - "scale": float scale factor applied
            - "pad_x": int left padding
            - "pad_y": int top padding
    """
    # Step 1: Optional deskew
    image = deskew(image)

    # Step 2: Resize with aspect-ratio-preserving padding
    resized, scale, pad_x, pad_y = resize_with_aspect_ratio(image, target_size)

    # Step 3: Mode-specific transform
    if mode == "recognition":
        resized = to_grayscale(resized)
    elif mode == "binary":
        resized = adaptive_binarize(resized)
    # else "detection" → keep RGB

    return {
        "image": resized,
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
    }


def scale_boxes(boxes, scale, pad_x, pad_y):
    """
    Transforms original bounding-box coordinates to match the preprocessed
    (resized + padded) image coordinate space.

    Args:
        boxes (np.ndarray): Shape (N, 8) with original coordinates.
        scale (float): Scale factor from preprocessing.
        pad_x (int): Horizontal padding offset.
        pad_y (int): Vertical padding offset.

    Returns:
        np.ndarray: Transformed boxes of shape (N, 8).
    """
    if len(boxes) == 0:
        return boxes.copy()

    transformed = boxes.astype(np.float64).copy()
    # Even indices are x-coordinates, odd indices are y-coordinates
    for i in range(4):
        transformed[:, i * 2] = transformed[:, i * 2] * scale + pad_x
        transformed[:, i * 2 + 1] = transformed[:, i * 2 + 1] * scale + pad_y

    return transformed.astype(np.int32)


# ===========================================================================
#  Data Augmentation
# ===========================================================================

class ReceiptAugmentor:
    """
    Applies random data augmentations to receipt images for training.

    Augmentations include:
        - Random rotation (±5°)
        - Brightness jitter
        - Contrast jitter
        - Gaussian blur
        - Salt-and-pepper noise

    Usage:
        aug = ReceiptAugmentor(p=0.5)
        augmented_pil = aug(pil_image)
    """

    def __init__(self, p=0.5, max_rotation=5.0):
        """
        Args:
            p (float): Probability of applying each augmentation.
            max_rotation (float): Maximum rotation angle in degrees.
        """
        self.p = p
        self.max_rotation = max_rotation

    def __call__(self, image):
        """
        Args:
            image (PIL.Image.Image): Input image.

        Returns:
            PIL.Image.Image: Augmented image.
        """
        # Random rotation
        if random.random() < self.p:
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            image = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))

        # Brightness jitter
        if random.random() < self.p:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Brightness(image).enhance(factor)

        # Contrast jitter
        if random.random() < self.p:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Contrast(image).enhance(factor)

        # Gaussian blur
        if random.random() < self.p * 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # Salt-and-pepper noise
        if random.random() < self.p * 0.2:
            image = self._add_noise(image)

        return image

    @staticmethod
    def _add_noise(image, amount=0.01):
        """Adds salt-and-pepper noise."""
        arr = np.array(image)
        num_salt = int(amount * arr.size * 0.5)
        num_pepper = int(amount * arr.size * 0.5)

        # Salt
        coords = tuple(np.random.randint(0, dim, num_salt) for dim in arr.shape)
        arr[coords] = 255

        # Pepper
        coords = tuple(np.random.randint(0, dim, num_pepper) for dim in arr.shape)
        arr[coords] = 0

        return Image.fromarray(arr)


# ===========================================================================
#  Dataset Splitting
# ===========================================================================

def discover_samples(data_dir):
    """
    Discovers all valid image files in the dataset and checks that matching
    annotation files exist.

    Args:
        data_dir (str): Root dataset directory containing img/, box/, key/ sub-dirs.

    Returns:
        list[str]: Sorted list of image filenames that have matching annotations.

    Raises:
        FileNotFoundError: If the img/ directory does not exist.
    """
    img_dir = os.path.join(data_dir, "img")
    box_dir = os.path.join(data_dir, "box")
    key_dir = os.path.join(data_dir, "key")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(
            f"Image directory '{img_dir}' does not exist. "
            "Run 'python scripts/download_dataset.py' first."
        )

    all_images = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith(VALID_IMAGE_EXTENSIONS)
    )

    # Filter to only images that have matching box AND key annotation files
    valid = []
    for img_name in all_images:
        base = os.path.splitext(img_name)[0]
        # Box files may be .csv or .txt
        has_box = (
            os.path.exists(os.path.join(box_dir, f"{base}.csv"))
            or os.path.exists(os.path.join(box_dir, f"{base}.txt"))
        )
        has_key = os.path.exists(os.path.join(key_dir, f"{base}.json"))
        if has_box and has_key:
            valid.append(img_name)
        else:
            logger.debug("Skipping %s (box=%s, key=%s)", img_name, has_box, has_key)

    logger.info("Discovered %d valid samples out of %d images", len(valid), len(all_images))
    return valid


def split_dataset(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=RANDOM_SEED):
    """
    Splits the dataset into train / validation / test subsets.

    The split is deterministic for a given seed, ensuring reproducibility
    across all team members.

    Args:
        data_dir (str): Root dataset directory.
        train_ratio (float): Fraction for training (default 0.8).
        val_ratio (float): Fraction for validation (default 0.1).
        test_ratio (float): Fraction for testing (default 0.1).
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_files, val_files, test_files) — each a list[str] of filenames.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    all_images = discover_samples(data_dir)
    n = len(all_images)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_files = [all_images[i] for i in indices[:train_end]]
    val_files = [all_images[i] for i in indices[train_end:val_end]]
    test_files = [all_images[i] for i in indices[val_end:]]

    logger.info("Split: train=%d, val=%d, test=%d", len(train_files), len(val_files), len(test_files))
    return train_files, val_files, test_files


# ===========================================================================
#  PyTorch Dataset
# ===========================================================================

if HAS_TORCH:
    class ReceiptDataset(Dataset):
        """
        PyTorch Dataset for SROIE receipt images with bounding-box and
        key-information annotations.

        Each sample is a dict containing:
            - "image":          Preprocessed PIL.Image (or tensor if transform given)
            - "original_image": Original PIL.Image (unmodified)
            - "boxes":          np.ndarray (N, 8) — original coordinate space
            - "scaled_boxes":   np.ndarray (N, 8) — preprocessed coordinate space
            - "transcripts":    list[str] of N text transcriptions
            - "keys":           dict with company, date, address, total
            - "filename":       str image filename
            - "scale":          float scale factor
            - "pad_x":          int x-padding
            - "pad_y":          int y-padding
        """

        def __init__(self, data_dir, image_files, target_size=DEFAULT_TARGET_SIZE,
                     mode="detection", augment=False, augment_p=0.5):
            """
            Args:
                data_dir (str): Root dataset directory.
                image_files (list[str]): Filenames to include.
                target_size (tuple): (W, H) for preprocessing.
                mode (str): "detection", "recognition", or "binary".
                augment (bool): Whether to apply random augmentations.
                augment_p (float): Augmentation probability.
            """
            self.data_dir = data_dir
            self.image_files = list(image_files)
            self.target_size = target_size
            self.mode = mode
            self.augmentor = ReceiptAugmentor(p=augment_p) if augment else None

            self.img_dir = os.path.join(data_dir, "img")
            self.box_dir = os.path.join(data_dir, "box")
            self.key_dir = os.path.join(data_dir, "key")

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            img_name = self.image_files[idx]
            base_name = os.path.splitext(img_name)[0]

            # --- Load image ---
            img_path = os.path.join(self.img_dir, img_name)
            original_image = load_image(img_path)

            # --- Load annotations ---
            # Try .csv first, then .txt
            box_path = os.path.join(self.box_dir, f"{base_name}.csv")
            if not os.path.exists(box_path):
                box_path = os.path.join(self.box_dir, f"{base_name}.txt")
            boxes, transcripts = load_boxes(box_path)

            key_path = os.path.join(self.key_dir, f"{base_name}.json")
            keys = load_keys(key_path)

            # --- Augmentation (training only) ---
            aug_image = original_image.copy()
            if self.augmentor is not None:
                aug_image = self.augmentor(aug_image)

            # --- Preprocessing ---
            prep = preprocess_image(aug_image, target_size=self.target_size, mode=self.mode)

            # --- Scale boxes to match preprocessed image ---
            scaled_boxes = scale_boxes(boxes, prep["scale"], prep["pad_x"], prep["pad_y"])

            return {
                "image": prep["image"],
                "original_image": original_image,
                "boxes": boxes,
                "scaled_boxes": scaled_boxes,
                "transcripts": transcripts,
                "keys": keys,
                "filename": img_name,
                "scale": prep["scale"],
                "pad_x": prep["pad_x"],
                "pad_y": prep["pad_y"],
            }


    def collate_fn(batch):
        """
        Custom collate function for ReceiptDataset.

        Converts PIL images to tensors (B, 3, H, W) normalised to [0, 1].
        Keeps variable-length annotations as lists.

        Returns:
            dict with keys:
                - "images":          Tensor (B, 3, H, W) float32 in [0, 1]
                - "original_images": list[PIL.Image]
                - "boxes":           list[np.ndarray] — original coordinates
                - "scaled_boxes":    list[np.ndarray] — preprocessed coordinates
                - "transcripts":     list[list[str]]
                - "keys":            list[dict]
                - "filenames":       list[str]
                - "scales":          list[float]
                - "pad_xs":          list[int]
                - "pad_ys":          list[int]
        """
        images = []
        for item in batch:
            arr = np.array(item["image"], dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
            images.append(tensor)

        return {
            "images": torch.stack(images),
            "original_images": [item["original_image"] for item in batch],
            "boxes": [item["boxes"] for item in batch],
            "scaled_boxes": [item["scaled_boxes"] for item in batch],
            "transcripts": [item["transcripts"] for item in batch],
            "keys": [item["keys"] for item in batch],
            "filenames": [item["filename"] for item in batch],
            "scales": [item["scale"] for item in batch],
            "pad_xs": [item["pad_x"] for item in batch],
            "pad_ys": [item["pad_y"] for item in batch],
        }


    def get_dataloaders(data_dir, batch_size=8, target_size=DEFAULT_TARGET_SIZE,
                        mode="detection", num_workers=0, seed=RANDOM_SEED):
        """
        Convenience function to create train / val / test DataLoaders.

        Training set has augmentation enabled; val and test do not.

        Args:
            data_dir (str): Root dataset directory.
            batch_size (int): Batch size.
            target_size (tuple): (W, H) for preprocessing.
            mode (str): "detection", "recognition", or "binary".
            num_workers (int): DataLoader worker processes (0 = main thread).
            seed (int): Random seed.

        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        train_files, val_files, test_files = split_dataset(data_dir, seed=seed)

        train_ds = ReceiptDataset(data_dir, train_files, target_size, mode, augment=True)
        val_ds = ReceiptDataset(data_dir, val_files, target_size, mode, augment=False)
        test_ds = ReceiptDataset(data_dir, test_files, target_size, mode, augment=False)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers
        )

        return train_loader, val_loader, test_loader


# ===========================================================================
#  Standalone Execution
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    data_dir = "dataset"

    print("=" * 60)
    print("  SROIE Data Pipeline — Task 1 Verification")
    print("=" * 60)

    # 1. Discover & split
    train_files, val_files, test_files = split_dataset(data_dir)
    print(f"\nDataset Split (seed={RANDOM_SEED}):")
    print(f"  Train : {len(train_files)} samples")
    print(f"  Val   : {len(val_files)} samples")
    print(f"  Test  : {len(test_files)} samples")

    # 2. Load & preprocess a single sample
    sample_file = train_files[0]
    img = load_image(os.path.join(data_dir, "img", sample_file))
    print(f"\nSample: {sample_file}")
    print(f"  Original size: {img.size}")

    prep = preprocess_image(img, mode="detection")
    print(f"  Preprocessed size: {prep['image'].size}")
    print(f"  Scale factor: {prep['scale']:.4f}")
    print(f"  Padding: ({prep['pad_x']}, {prep['pad_y']})")

    # 3. Load annotations
    base = os.path.splitext(sample_file)[0]
    box_path = os.path.join(data_dir, "box", f"{base}.csv")
    if not os.path.exists(box_path):
        box_path = os.path.join(data_dir, "box", f"{base}.txt")
    boxes, transcripts = load_boxes(box_path)
    keys = load_keys(os.path.join(data_dir, "key", f"{base}.json"))
    print(f"  Bounding boxes: {len(boxes)}")
    print(f"  Transcripts: {len(transcripts)}")
    print(f"  Keys: {keys}")

    # 4. Test DataLoader (if torch available)
    if HAS_TORCH:
        train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=4)
        batch = next(iter(train_loader))
        print(f"\nDataLoader batch test:")
        print(f"  Images tensor shape: {batch['images'].shape}")
        print(f"  Images dtype: {batch['images'].dtype}")
        print(f"  Filenames: {batch['filenames']}")
        print(f"  Boxes count per sample: {[len(b) for b in batch['boxes']]}")
    else:
        print("\n[WARN] PyTorch not installed — DataLoader test skipped.")

    print("\n" + "=" * 60)
    print("  All checks passed!")
    print("=" * 60)
