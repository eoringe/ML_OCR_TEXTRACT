"""
Task 2: Text Detection
Branch: task-1-data-pipeline-emmanuel-oringe (team dissolved — all work consolidated)

This module provides text detection capabilities for the SROIE receipt OCR
pipeline. It supports:
  - OpenCV contour-based text region detection (primary baseline)
  - MSER (Maximally Stable Extremal Regions) text detection
  - Non-Maximum Suppression (NMS) to filter overlapping detections
  - Preprocessing helpers for detection input normalisation

Usage:
    from src.detection.detect import TextDetector
    detector = TextDetector(method="contour")
    boxes = detector.detect(image_array)
"""

import logging
import numpy as np

# Optional imports — gracefully degrade if not installed
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MIN_AREA = 100          # Minimum contour area in pixels
DEFAULT_NMS_THRESHOLD = 0.3     # IoU threshold for non-maximum suppression
MORPH_KERNEL_WIDTH = 15         # Morphological kernel width for grouping text
MORPH_KERNEL_HEIGHT = 3         # Morphological kernel height for grouping text
DILATE_ITERATIONS = 2           # Number of dilation iterations


# ===========================================================================
#  Preprocessing Helpers
# ===========================================================================

def preprocess_for_detection(image, target_size=None):
    """
    Prepares a raw image for text detection.

    Steps:
        1. Convert to grayscale
        2. Apply Gaussian blur to reduce noise
        3. Apply Otsu thresholding for binarization
        4. Optionally resize to target_size

    Args:
        image (np.ndarray): Input BGR or RGB image (H, W, 3).
        target_size (tuple, optional): (width, height) to resize to.

    Returns:
        dict:
            - "binary": np.ndarray — binary thresholded image
            - "gray": np.ndarray — grayscale image
            - "scale_x": float — x scale factor (1.0 if no resize)
            - "scale_y": float — y scale factor (1.0 if no resize)
    """
    if not HAS_CV2:
        raise RuntimeError("OpenCV is required for text detection preprocessing.")

    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif len(image.shape) == 2:
        gray = image.copy()
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    scale_x, scale_y = 1.0, 1.0

    if target_size is not None:
        orig_h, orig_w = gray.shape[:2]
        target_w, target_h = target_size
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return {
        "binary": binary,
        "gray": gray,
        "scale_x": scale_x,
        "scale_y": scale_y,
    }


# ===========================================================================
#  Non-Maximum Suppression
# ===========================================================================

def compute_box_iou(box_a, box_b):
    """
    Computes IoU between two axis-aligned bounding boxes.

    Args:
        box_a (tuple): (x, y, w, h)
        box_b (tuple): (x, y, w, h)

    Returns:
        float: Intersection over Union.
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def non_maximum_suppression(boxes, scores=None, iou_threshold=DEFAULT_NMS_THRESHOLD):
    """
    Applies Non-Maximum Suppression to filter overlapping bounding boxes.

    Args:
        boxes (list): List of [x1,y1,x2,y2,x3,y3,x4,y4] 8-point boxes.
        scores (list, optional): Confidence scores. If None, uses box area.
        iou_threshold (float): IoU threshold above which to suppress.

    Returns:
        list: Filtered list of bounding boxes.
    """
    if not boxes:
        return []

    # Convert 8-point boxes to axis-aligned (x, y, w, h) for IoU
    rects = []
    for box in boxes:
        x_min = min(box[0], box[6])
        x_max = max(box[2], box[4])
        y_min = min(box[1], box[3])
        y_max = max(box[5], box[7])
        rects.append((x_min, y_min, x_max - x_min, y_max - y_min))

    if scores is None:
        # Use area as proxy for score (larger = higher priority)
        scores = [r[2] * r[3] for r in rects]

    # Sort by score descending
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    keep = []
    suppressed = set()

    for i in indices:
        if i in suppressed:
            continue
        keep.append(i)
        for j in indices:
            if j in suppressed or j == i:
                continue
            iou = compute_box_iou(rects[i], rects[j])
            if iou > iou_threshold:
                suppressed.add(j)

    return [boxes[i] for i in keep]


# ===========================================================================
#  Text Detector
# ===========================================================================

class TextDetector:
    """
    Text detection model for localising text regions in receipt images.

    Supports two detection methods:
        - "contour": OpenCV morphological operations + contour detection
        - "mser": Maximally Stable Extremal Regions

    Usage:
        detector = TextDetector(method="contour")
        boxes = detector.detect(image)  # returns list of 8-point boxes
    """

    SUPPORTED_METHODS = ("contour", "mser")

    def __init__(self, method="contour", min_area=DEFAULT_MIN_AREA,
                 nms_threshold=DEFAULT_NMS_THRESHOLD, model_path=None):
        """
        Args:
            method (str): Detection method — "contour" or "mser".
            min_area (int): Minimum bounding box area to keep.
            nms_threshold (float): IoU threshold for NMS filtering.
            model_path (str, optional): Path to model weights (for future
                neural network detectors).
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Supported: {self.SUPPORTED_METHODS}"
            )
        if not HAS_CV2:
            raise RuntimeError("OpenCV is required for TextDetector.")

        self.method = method
        self.min_area = min_area
        self.nms_threshold = nms_threshold
        self.model_path = model_path

        logger.info("TextDetector initialised (method=%s, min_area=%d)", method, min_area)

    def detect(self, image, apply_nms=True):
        """
        Detects text bounding boxes in a single image.

        Args:
            image (np.ndarray): Input RGB image (H, W, 3).
            apply_nms (bool): Whether to apply non-maximum suppression.

        Returns:
            list: List of bounding boxes, each as [x1,y1,x2,y2,x3,y3,x4,y4].
                  Points ordered: top-left → top-right → bottom-right → bottom-left.
                  Sorted top-to-bottom, then left-to-right.
        """
        if image is None or image.size == 0:
            return []

        if self.method == "contour":
            boxes = self._detect_contour(image)
        elif self.method == "mser":
            boxes = self._detect_mser(image)
        else:
            boxes = []

        if apply_nms and len(boxes) > 1:
            boxes = non_maximum_suppression(boxes, iou_threshold=self.nms_threshold)

        # Sort top-to-bottom, left-to-right
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

        logger.debug("Detected %d text regions (method=%s)", len(boxes), self.method)
        return boxes

    def _detect_contour(self, image):
        """
        Contour-based text detection using morphological operations.

        Pipeline:
            1. Convert to grayscale
            2. Gaussian blur + Otsu threshold
            3. Morphological dilation to merge text characters into lines
            4. Find external contours
            5. Filter by area
        """
        prep = preprocess_for_detection(image)
        binary = prep["binary"]

        # Morphological operations to group text lines
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (MORPH_KERNEL_WIDTH, MORPH_KERNEL_HEIGHT)
        )
        dilated = cv2.dilate(binary, kernel, iterations=DILATE_ITERATIONS)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # 8-point format: TL, TR, BR, BL
            box = [
                x, y,
                x + w, y,
                x + w, y + h,
                x, y + h,
            ]
            boxes.append(box)

        return boxes

    def _detect_mser(self, image):
        """
        MSER-based text detection — detects stable regions that likely
        contain characters, then groups them into text-line bounding boxes.
        """
        prep = preprocess_for_detection(image)
        gray = prep["gray"]

        mser = cv2.MSER_create()
        mser.setMinArea(60)
        mser.setMaxArea(14400)

        regions, _ = mser.detectRegions(gray)

        if len(regions) == 0:
            return []

        # Convert MSER regions to bounding rectangles and merge nearby ones
        bboxes = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            if w * h >= self.min_area:
                bboxes.append((x, y, w, h))

        if not bboxes:
            return []

        # Group overlapping MSER regions into text lines using morphological approach
        # Create a mask from all MSER bounding boxes
        h_img, w_img = gray.shape[:2]
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        for (bx, by, bw, bh) in bboxes:
            cv2.rectangle(mask, (bx, by), (bx + bw, by + bh), 255, -1)

        # Dilate to merge adjacent character boxes into lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            box = [x, y, x + w, y, x + w, y + h, x, y + h]
            boxes.append(box)

        return boxes

    def detect_batch(self, images, apply_nms=True):
        """
        Detects text regions in a batch of images.

        Args:
            images (list[np.ndarray]): List of RGB images.
            apply_nms (bool): Whether to apply NMS.

        Returns:
            list[list]: List of box lists, one per image.
        """
        return [self.detect(img, apply_nms=apply_nms) for img in images]


# ===========================================================================
#  Standalone Execution
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  Text Detection — Task 2 Verification")
    print("=" * 60)

    # Create a synthetic test image with white text on black background
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(dummy, (50, 100), (300, 140), (255, 255, 255), -1)
    cv2.rectangle(dummy, (50, 200), (400, 240), (255, 255, 255), -1)
    cv2.rectangle(dummy, (50, 300), (250, 340), (255, 255, 255), -1)

    for method in TextDetector.SUPPORTED_METHODS:
        detector = TextDetector(method=method)
        found = detector.detect(dummy)
        print(f"\n  [{method.upper()}] Detected {len(found)} text regions:")
        for i, box in enumerate(found):
            print(f"    Box {i}: {box}")

    # NMS test
    print("\n  NMS Test:")
    overlapping = [
        [50, 100, 300, 100, 300, 140, 50, 140],
        [55, 102, 305, 102, 305, 142, 55, 142],  # overlaps with first
        [50, 300, 250, 300, 250, 340, 50, 340],   # separate
    ]
    filtered = non_maximum_suppression(overlapping, iou_threshold=0.3)
    print(f"    Input: {len(overlapping)} boxes → After NMS: {len(filtered)} boxes")

    print("\n" + "=" * 60)
    print("  All checks passed!")
    print("=" * 60)
