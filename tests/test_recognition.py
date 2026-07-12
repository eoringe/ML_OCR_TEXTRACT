"""
Unit tests for Task 3: Text Recognition
Run with: python -m pytest tests/test_recognition.py -v
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from src.recognition.recognize import (
    TextRecognizer,
    crop_box,
    preprocess_crop_for_recognition,
    HAS_EASYOCR,
    HAS_TESSERACT,
)


# ======================================================================
#  Test: Crop Box
# ======================================================================

@unittest.skipUnless(HAS_CV2, "OpenCV not installed")
class TestCropBox(unittest.TestCase):
    """Tests for crop_box()."""

    def test_valid_crop(self):
        image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        box = [10, 10, 100, 10, 100, 50, 10, 50]
        result = crop_box(image, box)
        self.assertIsNotNone(result)
        self.assertEqual(result.ndim, 3)

    def test_crop_with_padding(self):
        image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        box = [10, 10, 100, 10, 100, 50, 10, 50]
        result = crop_box(image, box, pad=5)
        self.assertIsNotNone(result)
        # With padding, crop should be slightly larger
        no_pad = crop_box(image, box, pad=0)
        self.assertGreaterEqual(result.shape[0], no_pad.shape[0])

    def test_crop_at_edge(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        box = [0, 0, 99, 0, 99, 99, 0, 99]
        result = crop_box(image, box)
        self.assertIsNotNone(result)

    def test_invalid_box_returns_none(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Box with negative dimensions
        result = crop_box(image, [50, 50, 50, 50, 50, 50, 50, 50], pad=0)
        # A zero-area box might return None or empty
        if result is not None:
            self.assertEqual(result.size, 0) if result.ndim > 0 else None

    def test_grayscale_image(self):
        image = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
        box = [10, 10, 100, 10, 100, 50, 10, 50]
        result = crop_box(image, box)
        self.assertIsNotNone(result)


# ======================================================================
#  Test: Preprocess Crop
# ======================================================================

@unittest.skipUnless(HAS_CV2, "OpenCV not installed")
class TestPreprocessCrop(unittest.TestCase):
    """Tests for preprocess_crop_for_recognition()."""

    def test_resizes_to_target_height(self):
        crop = np.random.randint(0, 255, (50, 200, 3), dtype=np.uint8)
        result = preprocess_crop_for_recognition(crop, target_height=32)
        self.assertEqual(result.shape[0], 32)

    def test_preserves_aspect_ratio(self):
        crop = np.random.randint(0, 255, (50, 200, 3), dtype=np.uint8)
        result = preprocess_crop_for_recognition(crop, target_height=32)
        expected_width = int(32 * (200 / 50))
        self.assertAlmostEqual(result.shape[1], expected_width, delta=2)

    def test_grayscale_input(self):
        crop = np.random.randint(0, 255, (50, 200), dtype=np.uint8)
        result = preprocess_crop_for_recognition(crop, target_height=32)
        self.assertEqual(result.shape[0], 32)
        self.assertEqual(result.ndim, 2)


# ======================================================================
#  Test: TextRecognizer
# ======================================================================

class TestTextRecognizer(unittest.TestCase):
    """Tests for TextRecognizer class."""

    def test_init_default(self):
        recognizer = TextRecognizer(use_easyocr=False, use_tesseract=False)
        self.assertEqual(recognizer.backend, "mock")

    def test_backend_selection_easyocr(self):
        recognizer = TextRecognizer(use_easyocr=True, use_tesseract=False)
        if HAS_EASYOCR:
            self.assertEqual(recognizer.backend, "easyocr")
        else:
            self.assertEqual(recognizer.backend, "mock")

    def test_backend_selection_tesseract(self):
        recognizer = TextRecognizer(use_easyocr=False, use_tesseract=True)
        if HAS_TESSERACT:
            self.assertEqual(recognizer.backend, "tesseract")
        else:
            self.assertEqual(recognizer.backend, "mock")

    def test_recognize_empty_boxes(self):
        recognizer = TextRecognizer(use_easyocr=False, use_tesseract=False)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = recognizer.recognize(image, [])
        self.assertEqual(result, [])

    @unittest.skipUnless(HAS_CV2, "OpenCV not installed")
    def test_recognize_mock_returns_strings(self):
        recognizer = TextRecognizer(use_easyocr=False, use_tesseract=False)
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        boxes = [[10, 10, 100, 10, 100, 50, 10, 50]]
        result = recognizer.recognize(image, boxes)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], str)
        self.assertEqual(result[0], "MockText")

    @unittest.skipUnless(HAS_CV2, "OpenCV not installed")
    def test_recognize_multiple_boxes(self):
        recognizer = TextRecognizer(use_easyocr=False, use_tesseract=False)
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        boxes = [
            [10, 10, 100, 10, 100, 50, 10, 50],
            [10, 100, 200, 100, 200, 140, 10, 140],
            [10, 200, 150, 200, 150, 240, 10, 240],
        ]
        result = recognizer.recognize(image, boxes)
        self.assertEqual(len(result), 3)

    @unittest.skipUnless(HAS_CV2, "OpenCV not installed")
    def test_recognize_batch(self):
        recognizer = TextRecognizer(use_easyocr=False, use_tesseract=False)
        images = [
            np.zeros((100, 200, 3), dtype=np.uint8),
            np.zeros((200, 300, 3), dtype=np.uint8),
        ]
        boxes_list = [
            [[10, 10, 90, 10, 90, 40, 10, 40]],
            [[10, 10, 100, 10, 100, 50, 10, 50], [10, 100, 100, 100, 100, 140, 10, 140]],
        ]
        results = recognizer.recognize_batch(images, boxes_list)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0]), 1)
        self.assertEqual(len(results[1]), 2)

    def test_languages_default(self):
        recognizer = TextRecognizer(use_easyocr=False, use_tesseract=False)
        self.assertEqual(recognizer.languages, ["en"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
