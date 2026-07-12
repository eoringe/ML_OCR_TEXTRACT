"""
Unit tests for Utility Helpers
Run with: python -m pytest tests/test_helpers.py -v
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

from src.utils.helpers import (
    calculate_iou,
    unscale_boxes,
    polygon_area,
    box_to_rect,
    draw_polygon_on_image,
)


# ======================================================================
#  Test: IoU Calculation
# ======================================================================

class TestCalculateIoU(unittest.TestCase):
    """Tests for calculate_iou()."""

    def test_identical_boxes(self):
        iou = calculate_iou((0, 0, 10, 10), (0, 0, 10, 10))
        self.assertAlmostEqual(iou, 1.0)

    def test_no_overlap(self):
        iou = calculate_iou((0, 0, 5, 5), (10, 10, 15, 15))
        self.assertAlmostEqual(iou, 0.0)

    def test_partial_overlap(self):
        iou = calculate_iou((0, 0, 10, 10), (5, 5, 15, 15))
        # Intersection: 5x5=25, Union: 100+100-25=175
        expected = 25.0 / 175.0
        self.assertAlmostEqual(iou, expected, places=4)

    def test_contained_box(self):
        iou = calculate_iou((0, 0, 20, 20), (5, 5, 15, 15))
        # Intersection: 10x10=100, Union: 400+100-100=400
        expected = 100.0 / 400.0
        self.assertAlmostEqual(iou, expected, places=4)

    def test_touching_boxes(self):
        iou = calculate_iou((0, 0, 10, 10), (10, 0, 20, 10))
        self.assertAlmostEqual(iou, 0.0)

    def test_zero_area_boxes(self):
        iou = calculate_iou((5, 5, 5, 5), (5, 5, 5, 5))
        self.assertAlmostEqual(iou, 0.0)

    def test_negative_coordinates(self):
        iou = calculate_iou((-10, -10, 0, 0), (-5, -5, 5, 5))
        self.assertGreater(iou, 0.0)


# ======================================================================
#  Test: Unscale Boxes
# ======================================================================

class TestUnscaleBoxes(unittest.TestCase):
    """Tests for unscale_boxes()."""

    def test_identity(self):
        boxes = np.array([[10, 20, 100, 20, 100, 50, 10, 50]], dtype=np.int32)
        result = unscale_boxes(boxes, scale=1.0, pad_x=0, pad_y=0)
        np.testing.assert_array_equal(result, boxes)

    def test_unscale_padding(self):
        scaled = np.array([[15, 30, 105, 30, 105, 60, 15, 60]], dtype=np.int32)
        result = unscale_boxes(scaled, scale=1.0, pad_x=5, pad_y=10)
        expected = np.array([[10, 20, 100, 20, 100, 50, 10, 50]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_unscale_scale_factor(self):
        scaled = np.array([[20, 40, 200, 40, 200, 100, 20, 100]], dtype=np.int32)
        result = unscale_boxes(scaled, scale=2.0, pad_x=0, pad_y=0)
        expected = np.array([[10, 20, 100, 20, 100, 50, 10, 50]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_unscale_combined(self):
        # scale=2.0, pad_x=5, pad_y=5
        # Original: (0,0,10,0,10,10,0,10)
        # Scaled: (0*2+5, 0*2+5, 10*2+5, ...) = (5,5,25,5,25,25,5,25)
        scaled = np.array([[5, 5, 25, 5, 25, 25, 5, 25]], dtype=np.int32)
        result = unscale_boxes(scaled, scale=2.0, pad_x=5, pad_y=5)
        expected = np.array([[0, 0, 10, 0, 10, 10, 0, 10]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_empty_boxes(self):
        boxes = np.zeros((0, 8), dtype=np.int32)
        result = unscale_boxes(boxes, scale=1.0, pad_x=0, pad_y=0)
        self.assertEqual(result.shape, (0, 8))

    def test_zero_scale_raises(self):
        boxes = np.array([[10, 20, 100, 20, 100, 50, 10, 50]], dtype=np.int32)
        with self.assertRaises(ValueError):
            unscale_boxes(boxes, scale=0, pad_x=0, pad_y=0)

    def test_inverse_of_scale_boxes(self):
        """Verify that unscale_boxes is the inverse of scale_boxes."""
        from src.data_pipeline.prep import scale_boxes

        original = np.array([
            [10, 20, 100, 20, 100, 50, 10, 50],
            [30, 40, 200, 40, 200, 80, 30, 80],
        ], dtype=np.int32)
        scale = 1.5
        pad_x = 10
        pad_y = 20

        scaled = scale_boxes(original, scale, pad_x, pad_y)
        recovered = unscale_boxes(scaled, scale, pad_x, pad_y)

        np.testing.assert_array_almost_equal(recovered, original, decimal=0)


# ======================================================================
#  Test: Polygon Area
# ======================================================================

class TestPolygonArea(unittest.TestCase):
    """Tests for polygon_area()."""

    def test_square(self):
        box = [0, 0, 10, 0, 10, 10, 0, 10]
        area = polygon_area(box)
        self.assertAlmostEqual(area, 100.0)

    def test_rectangle(self):
        box = [0, 0, 20, 0, 20, 10, 0, 10]
        area = polygon_area(box)
        self.assertAlmostEqual(area, 200.0)

    def test_zero_area_point(self):
        box = [5, 5, 5, 5, 5, 5, 5, 5]
        area = polygon_area(box)
        self.assertAlmostEqual(area, 0.0)

    def test_zero_area_line(self):
        box = [0, 0, 10, 0, 10, 0, 0, 0]
        area = polygon_area(box)
        self.assertAlmostEqual(area, 0.0)

    def test_numpy_array_input(self):
        box = np.array([0, 0, 10, 0, 10, 10, 0, 10])
        area = polygon_area(box)
        self.assertAlmostEqual(area, 100.0)

    def test_non_origin_box(self):
        box = [50, 100, 150, 100, 150, 140, 50, 140]
        area = polygon_area(box)
        self.assertAlmostEqual(area, 4000.0)


# ======================================================================
#  Test: Box to Rect
# ======================================================================

class TestBoxToRect(unittest.TestCase):
    """Tests for box_to_rect()."""

    def test_basic_conversion(self):
        box = [10, 20, 100, 20, 100, 50, 10, 50]
        result = box_to_rect(box)
        self.assertEqual(result, (10, 20, 100, 50))

    def test_origin_box(self):
        box = [0, 0, 10, 0, 10, 10, 0, 10]
        result = box_to_rect(box)
        self.assertEqual(result, (0, 0, 10, 10))


# ======================================================================
#  Test: Draw Polygon
# ======================================================================

@unittest.skipUnless(HAS_CV2, "OpenCV not installed")
class TestDrawPolygon(unittest.TestCase):
    """Tests for draw_polygon_on_image()."""

    def test_draws_on_image(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        box = [10, 10, 100, 10, 100, 50, 10, 50]
        result = draw_polygon_on_image(image, box, color=(0, 255, 0))
        self.assertEqual(result.shape, (100, 200, 3))
        # At least some non-zero pixels (the drawn polygon)
        self.assertGreater(result.sum(), 0)

    def test_draws_with_label(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        box = [10, 20, 100, 20, 100, 50, 10, 50]
        result = draw_polygon_on_image(image, box, label="Text", color=(255, 0, 0))
        self.assertGreater(result.sum(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
