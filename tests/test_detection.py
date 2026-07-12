"""
Unit tests for Task 2: Text Detection
Run with: python -m pytest tests/test_detection.py -v
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

from src.detection.detect import (
    TextDetector,
    preprocess_for_detection,
    non_maximum_suppression,
    compute_box_iou,
)


# ======================================================================
#  Test: Preprocessing
# ======================================================================

@unittest.skipUnless(HAS_CV2, "OpenCV not installed")
class TestPreprocessForDetection(unittest.TestCase):
    """Tests for preprocess_for_detection()."""

    def test_returns_binary_and_gray(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = preprocess_for_detection(image)
        self.assertIn("binary", result)
        self.assertIn("gray", result)
        self.assertEqual(result["binary"].shape, (100, 200))
        self.assertEqual(result["gray"].shape, (100, 200))

    def test_scale_factors_default_to_one(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = preprocess_for_detection(image)
        self.assertAlmostEqual(result["scale_x"], 1.0)
        self.assertAlmostEqual(result["scale_y"], 1.0)

    def test_resize_updates_scale_factors(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = preprocess_for_detection(image, target_size=(400, 200))
        self.assertAlmostEqual(result["scale_x"], 2.0, places=1)
        self.assertAlmostEqual(result["scale_y"], 2.0, places=1)

    def test_handles_grayscale_input(self):
        image = np.zeros((100, 200), dtype=np.uint8)
        result = preprocess_for_detection(image)
        self.assertEqual(result["binary"].shape, (100, 200))

    def test_raises_on_invalid_shape(self):
        image = np.zeros((100, 200, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            preprocess_for_detection(image)


# ======================================================================
#  Test: NMS
# ======================================================================

class TestNonMaximumSuppression(unittest.TestCase):
    """Tests for non_maximum_suppression()."""

    def test_empty_input(self):
        result = non_maximum_suppression([])
        self.assertEqual(result, [])

    def test_single_box_unchanged(self):
        boxes = [[10, 10, 100, 10, 100, 50, 10, 50]]
        result = non_maximum_suppression(boxes)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], boxes[0])

    def test_overlapping_boxes_reduced(self):
        boxes = [
            [10, 10, 100, 10, 100, 50, 10, 50],
            [12, 12, 102, 12, 102, 52, 12, 52],  # Nearly identical
        ]
        result = non_maximum_suppression(boxes, iou_threshold=0.3)
        self.assertEqual(len(result), 1)

    def test_non_overlapping_boxes_kept(self):
        boxes = [
            [10, 10, 50, 10, 50, 30, 10, 30],
            [200, 200, 300, 200, 300, 250, 200, 250],
        ]
        result = non_maximum_suppression(boxes, iou_threshold=0.3)
        self.assertEqual(len(result), 2)

    def test_with_custom_scores(self):
        boxes = [
            [10, 10, 100, 10, 100, 50, 10, 50],
            [12, 12, 102, 12, 102, 52, 12, 52],
        ]
        # Give second box higher score
        scores = [1.0, 10.0]
        result = non_maximum_suppression(boxes, scores=scores, iou_threshold=0.3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], boxes[1])  # Higher score wins


# ======================================================================
#  Test: IoU
# ======================================================================

class TestComputeBoxIoU(unittest.TestCase):
    """Tests for compute_box_iou()."""

    def test_identical_boxes(self):
        iou = compute_box_iou((0, 0, 10, 10), (0, 0, 10, 10))
        self.assertAlmostEqual(iou, 1.0)

    def test_non_overlapping(self):
        iou = compute_box_iou((0, 0, 10, 10), (20, 20, 30, 30))
        self.assertAlmostEqual(iou, 0.0)

    def test_partial_overlap(self):
        iou = compute_box_iou((0, 0, 10, 10), (5, 5, 15, 15))
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

    def test_contained_box(self):
        iou = compute_box_iou((0, 0, 20, 20), (5, 5, 15, 15))
        self.assertGreater(iou, 0.0)

    def test_zero_area(self):
        iou = compute_box_iou((0, 0, 0, 0), (0, 0, 0, 0))
        self.assertAlmostEqual(iou, 0.0)


# ======================================================================
#  Test: TextDetector
# ======================================================================

@unittest.skipUnless(HAS_CV2, "OpenCV not installed")
class TestTextDetector(unittest.TestCase):
    """Tests for TextDetector class."""

    def test_init_contour_method(self):
        detector = TextDetector(method="contour")
        self.assertEqual(detector.method, "contour")

    def test_init_mser_method(self):
        detector = TextDetector(method="mser")
        self.assertEqual(detector.method, "mser")

    def test_init_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            TextDetector(method="invalid")

    def test_detect_empty_image(self):
        detector = TextDetector(method="contour")
        empty = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = detector.detect(empty)
        self.assertIsInstance(boxes, list)

    def test_detect_returns_list(self):
        detector = TextDetector(method="contour")
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        boxes = detector.detect(image)
        self.assertIsInstance(boxes, list)

    def test_detect_synthetic_image_contour(self):
        detector = TextDetector(method="contour", min_area=50)
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        # Draw white rectangles (simulating text regions)
        cv2.rectangle(image, (50, 100), (300, 140), (255, 255, 255), -1)
        cv2.rectangle(image, (50, 200), (400, 240), (255, 255, 255), -1)
        boxes = detector.detect(image)
        self.assertGreater(len(boxes), 0)

    def test_detect_synthetic_image_mser(self):
        detector = TextDetector(method="mser", min_area=50)
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 100), (300, 140), (255, 255, 255), -1)
        boxes = detector.detect(image)
        self.assertIsInstance(boxes, list)

    def test_box_format_is_8_points(self):
        detector = TextDetector(method="contour", min_area=50)
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 100), (300, 140), (255, 255, 255), -1)
        boxes = detector.detect(image)
        if boxes:
            self.assertEqual(len(boxes[0]), 8)

    def test_detect_sorted_top_to_bottom(self):
        detector = TextDetector(method="contour", min_area=50)
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 300), (300, 340), (255, 255, 255), -1)
        cv2.rectangle(image, (50, 100), (300, 140), (255, 255, 255), -1)
        boxes = detector.detect(image)
        if len(boxes) >= 2:
            # First box should be above second (lower y value)
            self.assertLessEqual(boxes[0][1], boxes[1][1])

    def test_detect_none_image(self):
        detector = TextDetector(method="contour")
        boxes = detector.detect(None)
        self.assertEqual(boxes, [])

    def test_detect_batch(self):
        detector = TextDetector(method="contour")
        images = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((200, 200, 3), dtype=np.uint8),
        ]
        results = detector.detect_batch(images)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], list)
        self.assertIsInstance(results[1], list)

    def test_detect_without_nms(self):
        detector = TextDetector(method="contour", min_area=50)
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 100), (300, 140), (255, 255, 255), -1)
        boxes = detector.detect(image, apply_nms=False)
        self.assertIsInstance(boxes, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
