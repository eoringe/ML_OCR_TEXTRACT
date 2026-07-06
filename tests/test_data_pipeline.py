"""
Unit tests for Task 1: Data Preprocessing & Loader
Author: Emmanuel Oringe

Run with:
    python -m pytest tests/test_data_pipeline.py -v
    OR
    python -m unittest tests.test_data_pipeline -v
"""

import os
import sys
import json
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_pipeline.prep import (
    load_boxes,
    load_keys,
    load_image,
    resize_with_aspect_ratio,
    to_grayscale,
    binarize,
    adaptive_binarize,
    preprocess_image,
    scale_boxes,
    ReceiptAugmentor,
    discover_samples,
    split_dataset,
    SROIE_FIELDS,
    DEFAULT_TARGET_SIZE,
)

# Check optional dependencies
try:
    import torch
    from src.data_pipeline.prep import ReceiptDataset, collate_fn, get_dataloaders
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class _MockDatasetMixin:
    """Mixin that creates a temporary mock SROIE dataset for testing."""

    NUM_SAMPLES = 10

    @classmethod
    def _create_mock_dataset(cls, tmp_dir):
        """Creates a minimal SROIE-format dataset in tmp_dir."""
        img_dir = os.path.join(tmp_dir, "img")
        box_dir = os.path.join(tmp_dir, "box")
        key_dir = os.path.join(tmp_dir, "key")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(box_dir, exist_ok=True)
        os.makedirs(key_dir, exist_ok=True)

        for i in range(cls.NUM_SAMPLES):
            name = f"{i:03d}"
            # Create a random RGB image of varying sizes
            w = np.random.randint(200, 600)
            h = np.random.randint(400, 1200)
            arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            img = Image.fromarray(arr, "RGB")
            img.save(os.path.join(img_dir, f"{name}.jpg"))

            # Create box annotation (.csv format as in the real SROIE dataset)
            num_boxes = np.random.randint(3, 8)
            lines = []
            for _ in range(num_boxes):
                x1 = np.random.randint(10, w // 2)
                y1 = np.random.randint(10, h // 2)
                bw = np.random.randint(50, min(200, w - x1))
                bh = np.random.randint(10, min(40, h - y1))
                x2, y2 = x1 + bw, y1
                x3, y3 = x1 + bw, y1 + bh
                x4, y4 = x1, y1 + bh
                text = f"SAMPLE TEXT {np.random.randint(100, 999)}"
                lines.append(f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{text}")
            with open(os.path.join(box_dir, f"{name}.csv"), "w") as f:
                f.write("\n".join(lines))

            # Create key annotation
            keys = {
                "company": f"COMPANY_{name}",
                "date": f"0{i+1}/01/2024",
                "address": f"{i*10} MAIN STREET",
                "total": f"{(i+1) * 5.50:.2f}",
            }
            with open(os.path.join(key_dir, f"{name}.json"), "w") as f:
                json.dump(keys, f)

        return tmp_dir


# ======================================================================
#  Test: Annotation Loaders
# ======================================================================

class TestLoadBoxes(unittest.TestCase):
    """Tests for load_boxes()."""

    def test_nonexistent_file_returns_empty(self):
        boxes, transcripts = load_boxes("/nonexistent/path.csv")
        self.assertEqual(boxes.shape, (0, 8))
        self.assertEqual(transcripts, [])

    def test_loads_valid_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("10,20,100,20,100,50,10,50,HELLO WORLD\n")
            f.write("15,60,200,60,200,90,15,90,SECOND LINE\n")
            path = f.name
        try:
            boxes, transcripts = load_boxes(path)
            self.assertEqual(boxes.shape, (2, 8))
            self.assertEqual(len(transcripts), 2)
            self.assertEqual(transcripts[0], "HELLO WORLD")
            self.assertEqual(transcripts[1], "SECOND LINE")
            np.testing.assert_array_equal(boxes[0], [10, 20, 100, 20, 100, 50, 10, 50])
        finally:
            os.unlink(path)

    def test_handles_commas_in_transcript(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("10,20,100,20,100,50,10,50,NO.53 55,57 & 59, JALAN SAGU\n")
            path = f.name
        try:
            boxes, transcripts = load_boxes(path)
            self.assertEqual(len(transcripts), 1)
            self.assertEqual(transcripts[0], "NO.53 55,57 & 59, JALAN SAGU")
        finally:
            os.unlink(path)

    def test_skips_malformed_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("bad line\n")
            f.write("\n")
            f.write("10,20,100,20,100,50,10,50,GOOD LINE\n")
            path = f.name
        try:
            boxes, transcripts = load_boxes(path)
            self.assertEqual(len(transcripts), 1)
            self.assertEqual(transcripts[0], "GOOD LINE")
        finally:
            os.unlink(path)

    def test_boxes_dtype_is_int32(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("10.5,20.3,100,20,100,50,10,50,TEXT\n")
            path = f.name
        try:
            boxes, _ = load_boxes(path)
            self.assertEqual(boxes.dtype, np.int32)
        finally:
            os.unlink(path)


class TestLoadKeys(unittest.TestCase):
    """Tests for load_keys()."""

    def test_nonexistent_file_returns_empty_keys(self):
        keys = load_keys("/nonexistent/path.json")
        for field in SROIE_FIELDS:
            self.assertIn(field, keys)
            self.assertEqual(keys[field], "")

    def test_loads_valid_json(self):
        data = {"company": "ACME", "date": "01/01/2024", "address": "123 ST", "total": "42.00"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            keys = load_keys(path)
            self.assertEqual(keys["company"], "ACME")
            self.assertEqual(keys["total"], "42.00")
        finally:
            os.unlink(path)

    def test_handles_missing_fields(self):
        data = {"company": "ACME"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            keys = load_keys(path)
            self.assertEqual(keys["company"], "ACME")
            self.assertEqual(keys["date"], "")
            self.assertEqual(keys["address"], "")
            self.assertEqual(keys["total"], "")
        finally:
            os.unlink(path)

    def test_handles_corrupt_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{bad json")
            path = f.name
        try:
            keys = load_keys(path)
            for field in SROIE_FIELDS:
                self.assertEqual(keys[field], "")
        finally:
            os.unlink(path)


# ======================================================================
#  Test: Image Loading
# ======================================================================

class TestLoadImage(unittest.TestCase):
    """Tests for load_image()."""

    def test_loads_valid_image(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.new("RGB", (100, 200), color=(128, 64, 32))
            img.save(f.name)
            path = f.name
        try:
            loaded = load_image(path)
            self.assertIsInstance(loaded, Image.Image)
            self.assertEqual(loaded.mode, "RGB")
            self.assertEqual(loaded.size, (100, 200))
        finally:
            os.unlink(path)

    def test_raises_on_nonexistent(self):
        with self.assertRaises(FileNotFoundError):
            load_image("/definitely/not/a/real/path.jpg")


# ======================================================================
#  Test: Image Preprocessing Functions
# ======================================================================

class TestResizeWithAspectRatio(unittest.TestCase):
    """Tests for resize_with_aspect_ratio()."""

    def test_output_size_matches_target(self):
        img = Image.new("RGB", (300, 600))
        resized, scale, pad_x, pad_y = resize_with_aspect_ratio(img, (640, 640))
        self.assertEqual(resized.size, (640, 640))

    def test_preserves_aspect_ratio(self):
        img = Image.new("RGB", (200, 400))
        resized, scale, pad_x, pad_y = resize_with_aspect_ratio(img, (640, 640))
        # The image is taller than wide, so height should dominate
        # scale = 640/400 = 1.6, new_w = 200*1.6=320, new_h = 400*1.6=640
        self.assertAlmostEqual(scale, 1.6, places=2)
        self.assertEqual(pad_y, 0)   # No vertical padding (height fills)
        self.assertEqual(pad_x, 160) # Horizontal padding = (640-320)//2

    def test_square_image(self):
        img = Image.new("RGB", (640, 640))
        resized, scale, pad_x, pad_y = resize_with_aspect_ratio(img, (640, 640))
        self.assertAlmostEqual(scale, 1.0, places=5)
        self.assertEqual(pad_x, 0)
        self.assertEqual(pad_y, 0)

    def test_wide_image(self):
        img = Image.new("RGB", (1000, 200))
        resized, scale, pad_x, pad_y = resize_with_aspect_ratio(img, (640, 640))
        # Width-dominant: scale = 640/1000 = 0.64
        self.assertAlmostEqual(scale, 0.64, places=2)
        self.assertTrue(pad_y > 0)  # Vertical padding needed


class TestToGrayscale(unittest.TestCase):
    """Tests for to_grayscale()."""

    def test_returns_3_channel_gray(self):
        img = Image.new("RGB", (100, 100), color=(128, 64, 32))
        gray = to_grayscale(img)
        self.assertEqual(gray.mode, "RGB")
        self.assertEqual(gray.size, (100, 100))
        # All three channels should be identical
        r, g, b = gray.split()
        self.assertEqual(list(r.getdata()), list(g.getdata()))
        self.assertEqual(list(g.getdata()), list(b.getdata()))


class TestBinarize(unittest.TestCase):
    """Tests for binarize()."""

    def test_output_is_binary(self):
        img = Image.new("RGB", (50, 50), color=(100, 100, 100))
        result = binarize(img, threshold=128)
        arr = np.array(result)
        unique = set(np.unique(arr))
        self.assertTrue(unique.issubset({0, 255}))

    def test_returns_3_channel(self):
        img = Image.new("RGB", (50, 50))
        result = binarize(img)
        self.assertEqual(result.mode, "RGB")


class TestPreprocessImage(unittest.TestCase):
    """Tests for preprocess_image()."""

    def test_detection_mode_returns_rgb(self):
        img = Image.new("RGB", (300, 500))
        result = preprocess_image(img, target_size=(320, 320), mode="detection")
        self.assertIn("image", result)
        self.assertEqual(result["image"].size, (320, 320))
        self.assertEqual(result["image"].mode, "RGB")

    def test_recognition_mode_returns_grayscale(self):
        img = Image.new("RGB", (300, 500), color=(200, 100, 50))
        result = preprocess_image(img, target_size=(320, 320), mode="recognition")
        # Should be gray (3-channel but all channels identical)
        r, g, b = result["image"].split()
        self.assertEqual(list(r.getdata()), list(g.getdata()))

    def test_binary_mode_returns_binary(self):
        img = Image.new("RGB", (300, 500), color=(120, 120, 120))
        result = preprocess_image(img, target_size=(320, 320), mode="binary")
        arr = np.array(result["image"])
        unique = set(np.unique(arr))
        self.assertTrue(unique.issubset({0, 255}))

    def test_returns_scale_and_padding(self):
        img = Image.new("RGB", (300, 500))
        result = preprocess_image(img, target_size=(640, 640))
        self.assertIn("scale", result)
        self.assertIn("pad_x", result)
        self.assertIn("pad_y", result)
        self.assertIsInstance(result["scale"], float)
        self.assertGreater(result["scale"], 0)


# ======================================================================
#  Test: Bounding Box Scaling
# ======================================================================

class TestScaleBoxes(unittest.TestCase):
    """Tests for scale_boxes()."""

    def test_empty_boxes(self):
        boxes = np.zeros((0, 8), dtype=np.int32)
        result = scale_boxes(boxes, 1.0, 0, 0)
        self.assertEqual(result.shape, (0, 8))

    def test_identity_transform(self):
        boxes = np.array([[10, 20, 100, 20, 100, 50, 10, 50]], dtype=np.int32)
        result = scale_boxes(boxes, 1.0, 0, 0)
        np.testing.assert_array_equal(result, boxes)

    def test_scaling(self):
        boxes = np.array([[10, 20, 100, 20, 100, 50, 10, 50]], dtype=np.int32)
        result = scale_boxes(boxes, 2.0, 0, 0)
        expected = np.array([[20, 40, 200, 40, 200, 100, 20, 100]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_scaling_with_padding(self):
        boxes = np.array([[10, 20, 100, 20, 100, 50, 10, 50]], dtype=np.int32)
        result = scale_boxes(boxes, 1.0, 5, 10)
        expected = np.array([[15, 30, 105, 30, 105, 60, 15, 60]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_multiple_boxes(self):
        boxes = np.array([
            [0, 0, 10, 0, 10, 10, 0, 10],
            [20, 20, 30, 20, 30, 30, 20, 30],
        ], dtype=np.int32)
        result = scale_boxes(boxes, 2.0, 5, 5)
        self.assertEqual(result.shape, (2, 8))
        # First box: x*2+5, y*2+5 → (5,5, 25,5, 25,25, 5,25)
        np.testing.assert_array_equal(result[0], [5, 5, 25, 5, 25, 25, 5, 25])


# ======================================================================
#  Test: Data Augmentation
# ======================================================================

class TestReceiptAugmentor(unittest.TestCase):
    """Tests for ReceiptAugmentor."""

    def test_returns_pil_image(self):
        aug = ReceiptAugmentor(p=1.0)
        img = Image.new("RGB", (200, 300), color=(128, 128, 128))
        result = aug(img)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (200, 300))

    def test_no_augmentation_when_p_zero(self):
        aug = ReceiptAugmentor(p=0.0)
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        result = aug(img)
        # With p=0, no augmentation should be applied
        np.testing.assert_array_equal(np.array(img), np.array(result))

    def test_augmentation_changes_image(self):
        np.random.seed(99)
        random_module = __import__("random")
        random_module.seed(99)
        aug = ReceiptAugmentor(p=1.0)
        img = Image.new("RGB", (200, 300), color=(128, 128, 128))
        result = aug(img)
        # With p=1.0, at least one augmentation should modify the image
        # (rotation, brightness, or contrast will change pixel values)
        self.assertFalse(
            np.array_equal(np.array(img), np.array(result)),
            "Augmentation with p=1.0 should change the image"
        )


# ======================================================================
#  Test: Dataset Discovery & Splitting
# ======================================================================

class TestDiscoverAndSplit(_MockDatasetMixin, unittest.TestCase):
    """Tests for discover_samples() and split_dataset()."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._create_mock_dataset(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_discover_finds_all_samples(self):
        samples = discover_samples(self.tmp_dir)
        self.assertEqual(len(samples), self.NUM_SAMPLES)

    def test_discover_returns_sorted_list(self):
        samples = discover_samples(self.tmp_dir)
        self.assertEqual(samples, sorted(samples))

    def test_discover_raises_on_missing_dir(self):
        with self.assertRaises(FileNotFoundError):
            discover_samples("/nonexistent/dataset")

    def test_discover_skips_images_without_annotations(self):
        # Add an image without corresponding box/key files
        extra_img = Image.new("RGB", (100, 100))
        extra_img.save(os.path.join(self.tmp_dir, "img", "orphan.jpg"))
        samples = discover_samples(self.tmp_dir)
        self.assertEqual(len(samples), self.NUM_SAMPLES)  # orphan should be skipped
        self.assertNotIn("orphan.jpg", samples)

    def test_split_ratios(self):
        train, val, test = split_dataset(self.tmp_dir, 0.8, 0.1, 0.1)
        total = len(train) + len(val) + len(test)
        self.assertEqual(total, self.NUM_SAMPLES)
        self.assertEqual(len(train), 8)  # 80% of 10
        self.assertEqual(len(val), 1)    # 10% of 10
        self.assertEqual(len(test), 1)   # 10% of 10

    def test_split_is_deterministic(self):
        t1, v1, te1 = split_dataset(self.tmp_dir, seed=42)
        t2, v2, te2 = split_dataset(self.tmp_dir, seed=42)
        self.assertEqual(t1, t2)
        self.assertEqual(v1, v2)
        self.assertEqual(te1, te2)

    def test_split_different_seeds_differ(self):
        t1, _, _ = split_dataset(self.tmp_dir, seed=42)
        t2, _, _ = split_dataset(self.tmp_dir, seed=99)
        self.assertNotEqual(t1, t2)

    def test_no_overlap_between_splits(self):
        train, val, test = split_dataset(self.tmp_dir)
        all_files = train + val + test
        self.assertEqual(len(all_files), len(set(all_files)), "Splits should not overlap")


# ======================================================================
#  Test: PyTorch Dataset & DataLoader (skip if torch not installed)
# ======================================================================

@unittest.skipUnless(HAS_TORCH, "PyTorch not installed — skipping Dataset tests")
class TestReceiptDataset(_MockDatasetMixin, unittest.TestCase):
    """Tests for ReceiptDataset and collate_fn."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._create_mock_dataset(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_dataset_len(self):
        samples = discover_samples(self.tmp_dir)
        ds = ReceiptDataset(self.tmp_dir, samples, target_size=(320, 320))
        self.assertEqual(len(ds), self.NUM_SAMPLES)

    def test_getitem_returns_expected_keys(self):
        samples = discover_samples(self.tmp_dir)
        ds = ReceiptDataset(self.tmp_dir, samples, target_size=(320, 320))
        item = ds[0]
        expected_keys = {
            "image", "original_image", "boxes", "scaled_boxes",
            "transcripts", "keys", "filename", "scale", "pad_x", "pad_y"
        }
        self.assertEqual(set(item.keys()), expected_keys)

    def test_image_is_pil_and_correct_size(self):
        samples = discover_samples(self.tmp_dir)
        ds = ReceiptDataset(self.tmp_dir, samples, target_size=(320, 320))
        item = ds[0]
        self.assertIsInstance(item["image"], Image.Image)
        self.assertEqual(item["image"].size, (320, 320))

    def test_boxes_shape(self):
        samples = discover_samples(self.tmp_dir)
        ds = ReceiptDataset(self.tmp_dir, samples, target_size=(320, 320))
        item = ds[0]
        self.assertEqual(item["boxes"].ndim, 2)
        self.assertEqual(item["boxes"].shape[1], 8)
        self.assertEqual(item["scaled_boxes"].shape, item["boxes"].shape)

    def test_collate_fn_produces_tensor(self):
        samples = discover_samples(self.tmp_dir)
        ds = ReceiptDataset(self.tmp_dir, samples[:3], target_size=(320, 320))
        batch = collate_fn([ds[i] for i in range(3)])
        self.assertEqual(batch["images"].shape, (3, 3, 320, 320))
        self.assertEqual(batch["images"].dtype, torch.float32)
        self.assertTrue((batch["images"] >= 0).all() and (batch["images"] <= 1).all())

    def test_dataloader_iteration(self):
        train_loader, val_loader, test_loader = get_dataloaders(
            self.tmp_dir, batch_size=4, target_size=(320, 320)
        )
        batch = next(iter(train_loader))
        self.assertIn("images", batch)
        self.assertEqual(batch["images"].dim(), 4)
        # Batch size <= 4 (last batch might be smaller)
        self.assertLessEqual(batch["images"].shape[0], 4)

    def test_augmentation_flag(self):
        samples = discover_samples(self.tmp_dir)
        ds_aug = ReceiptDataset(self.tmp_dir, samples, augment=True, augment_p=1.0)
        ds_no = ReceiptDataset(self.tmp_dir, samples, augment=False)
        self.assertIsNotNone(ds_aug.augmentor)
        self.assertIsNone(ds_no.augmentor)


# ======================================================================
#  Test: End-to-End with Real Dataset (only if dataset exists)
# ======================================================================

REAL_DATASET = os.path.join(PROJECT_ROOT, "dataset")

@unittest.skipUnless(os.path.isdir(os.path.join(REAL_DATASET, "img")), "Real dataset not available")
class TestRealDataset(unittest.TestCase):
    """Integration tests using the actual SROIE dataset on disk."""

    def test_discover_real_samples(self):
        samples = discover_samples(REAL_DATASET)
        self.assertGreater(len(samples), 500, "Should have 600+ real samples")

    def test_load_real_box_file(self):
        box_path = os.path.join(REAL_DATASET, "box", "000.csv")
        boxes, transcripts = load_boxes(box_path)
        self.assertGreater(len(boxes), 0)
        self.assertEqual(len(boxes), len(transcripts))
        self.assertEqual(boxes.shape[1], 8)

    def test_load_real_key_file(self):
        key_path = os.path.join(REAL_DATASET, "key", "000.json")
        keys = load_keys(key_path)
        self.assertNotEqual(keys["company"], "")
        self.assertNotEqual(keys["total"], "")

    def test_preprocess_real_image(self):
        img = load_image(os.path.join(REAL_DATASET, "img", "000.jpg"))
        result = preprocess_image(img, target_size=(640, 640), mode="detection")
        self.assertEqual(result["image"].size, (640, 640))
        self.assertGreater(result["scale"], 0)

    def test_split_real_dataset(self):
        train, val, test = split_dataset(REAL_DATASET)
        total = len(train) + len(val) + len(test)
        self.assertGreater(total, 500)
        # No overlap
        all_f = set(train) | set(val) | set(test)
        self.assertEqual(len(all_f), total)


if __name__ == "__main__":
    unittest.main(verbosity=2)
