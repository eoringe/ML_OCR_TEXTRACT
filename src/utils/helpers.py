"""
Utility helpers for the SROIE receipt OCR pipeline.

Provides:
  - IoU (Intersection over Union) calculation
  - Polygon drawing on images
  - Box coordinate transformations (unscale)
  - Polygon area calculation
  - Directory structure creation

Usage:
    from src.utils.helpers import calculate_iou, unscale_boxes, polygon_area
"""

import os
import logging
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

logger = logging.getLogger(__name__)


# ===========================================================================
#  Directory Utilities
# ===========================================================================

def create_directory_structure(root_path):
    """
    Ensures that all key directories for the OCR Textract pipeline are created.

    Args:
        root_path (str): Project root directory.
    """
    dirs = [
        "dataset",
        "dataset/img",
        "dataset/box",
        "dataset/key",
        "src",
        "src/data_pipeline",
        "src/detection",
        "src/recognition",
        "src/kie",
        "src/post_processing",
        "src/backend",
        "src/frontend",
        "src/utils",
        "scripts",
        "checkpoints",
    ]
    for d in dirs:
        path = os.path.join(root_path, d)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")


# ===========================================================================
#  IoU Calculation
# ===========================================================================

def calculate_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) of two axis-aligned bounding boxes.

    Args:
        box1 (tuple): (x1, y1, x2, y2) — top-left and bottom-right corners.
        box2 (tuple): (x1, y1, x2, y2) — top-left and bottom-right corners.

    Returns:
        float: IoU value in [0, 1].
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area

    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area


# ===========================================================================
#  Polygon / Bounding Box Utilities
# ===========================================================================

def draw_polygon_on_image(image, box, label=None, color=(0, 255, 0), thickness=2):
    """
    Draws an 8-point bounding box polygon on an image.

    Args:
        image (np.ndarray): Image to draw on (will be modified in place).
        box (list): 8-point bounding box [x1,y1,x2,y2,x3,y3,x4,y4].
        label (str, optional): Text label to draw above the box.
        color (tuple): BGR colour for the polygon.
        thickness (int): Line thickness.

    Returns:
        np.ndarray: Image with polygon drawn.
    """
    if not HAS_CV2:
        logger.warning("OpenCV not available — cannot draw polygons.")
        return image

    pts = np.array(box).reshape((-1, 2)).astype(np.int32)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
    if label:
        cv2.putText(
            image, label, (pts[0][0], pts[0][1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1
        )
    return image


def unscale_boxes(scaled_boxes, scale, pad_x, pad_y):
    """
    Inverse of scale_boxes — maps preprocessed coordinates back to original
    image space.

    This is useful for converting detection predictions (in the resized+padded
    coordinate space) back to the original image dimensions.

    Args:
        scaled_boxes (np.ndarray): Shape (N, 8) with preprocessed coordinates.
        scale (float): Scale factor that was applied during preprocessing.
        pad_x (int): Horizontal padding that was applied.
        pad_y (int): Vertical padding that was applied.

    Returns:
        np.ndarray: Unscaled boxes of shape (N, 8) in original coordinates.
    """
    if len(scaled_boxes) == 0:
        return scaled_boxes.copy()

    if scale == 0:
        raise ValueError("Scale factor cannot be zero.")

    result = scaled_boxes.astype(np.float64).copy()
    for i in range(4):
        result[:, i * 2] = (result[:, i * 2] - pad_x) / scale
        result[:, i * 2 + 1] = (result[:, i * 2 + 1] - pad_y) / scale

    return result.astype(np.int32)


def polygon_area(box):
    """
    Computes the area of a polygon defined by an 8-point bounding box
    using the Shoelace formula.

    Args:
        box (list or np.ndarray): 8 values [x1,y1,x2,y2,x3,y3,x4,y4].

    Returns:
        float: Area of the polygon.
    """
    pts = np.array(box).reshape((-1, 2)).astype(np.float64)
    n = len(pts)

    # Shoelace formula
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]

    return abs(area) / 2.0


def box_to_rect(box):
    """
    Converts an 8-point bounding box to an axis-aligned rectangle.

    Args:
        box (list): [x1,y1,x2,y2,x3,y3,x4,y4]

    Returns:
        tuple: (x_min, y_min, x_max, y_max)
    """
    pts = np.array(box).reshape((-1, 2))
    x_min = pts[:, 0].min()
    y_min = pts[:, 1].min()
    x_max = pts[:, 0].max()
    y_max = pts[:, 1].max()
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def merge_boxes_into_lines(boxes, transcripts, overlap_threshold=0.5):
    """
    Groups bounding boxes on the same horizontal line and merges their texts.
    Useful for connecting label & value boxes in OCR pipelines.
    """
    if not boxes or len(boxes) != len(transcripts):
        return boxes, transcripts

    items = []
    for idx, (box, text) in enumerate(zip(boxes, transcripts)):
        y_coords = [box[1], box[3], box[5], box[7]]
        x_coords = [box[0], box[2], box[4], box[6]]
        ymin, ymax = min(y_coords), max(y_coords)
        xmin, xmax = min(x_coords), max(x_coords)
        items.append({
            "idx": idx,
            "box": box,
            "text": text,
            "ymin": ymin,
            "ymax": ymax,
            "xmin": xmin,
            "xmax": xmax,
            "height": ymax - ymin,
            "y_center": (ymin + ymax) / 2.0
        })

    rows = []
    for item in sorted(items, key=lambda x: x["y_center"]):
        placed = False
        for row in rows:
            row_ymin = sum(r["ymin"] for r in row) / len(row)
            row_ymax = sum(r["ymax"] for r in row) / len(row)
            row_height = row_ymax - row_ymin

            overlap = min(item["ymax"], row_ymax) - max(item["ymin"], row_ymin)
            min_h = min(item["height"], row_height)

            if min_h > 0 and (overlap / min_h) > overlap_threshold:
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])

    rows.sort(key=lambda r: sum(item["y_center"] for item in r) / len(r))

    merged_boxes = []
    merged_transcripts = []

    for row in rows:
        row.sort(key=lambda x: x["xmin"])
        texts = [x["text"] for x in row if x["text"].strip()]
        if not texts:
            continue
        merged_text = " ".join(texts)

        x_min = min(x["xmin"] for x in row)
        x_max = max(x["xmax"] for x in row)
        y_min = min(x["ymin"] for x in row)
        y_max = max(x["ymax"] for x in row)

        merged_box = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
        merged_boxes.append(merged_box)
        merged_transcripts.append(merged_text)

    return merged_boxes, merged_transcripts


# ===========================================================================
#  Standalone Execution
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Utility Helpers — Verification")
    print("=" * 60)

    # IoU test
    print("\n  IoU Tests:")
    iou1 = calculate_iou((0, 0, 10, 10), (5, 5, 15, 15))
    print(f"    Overlapping boxes: IoU = {iou1:.4f}")
    iou2 = calculate_iou((0, 0, 10, 10), (20, 20, 30, 30))
    print(f"    Non-overlapping boxes: IoU = {iou2:.4f}")
    iou3 = calculate_iou((0, 0, 10, 10), (0, 0, 10, 10))
    print(f"    Identical boxes: IoU = {iou3:.4f}")

    # Unscale test
    print("\n  Unscale Boxes Test:")
    import numpy as np
    scaled = np.array([[25, 30, 105, 30, 105, 60, 25, 60]], dtype=np.int32)
    original = unscale_boxes(scaled, scale=1.0, pad_x=5, pad_y=10)
    print(f"    Scaled:   {scaled[0].tolist()}")
    print(f"    Unscaled: {original[0].tolist()}")

    # Polygon area test
    print("\n  Polygon Area Test:")
    box = [0, 0, 10, 0, 10, 10, 0, 10]
    area = polygon_area(box)
    print(f"    Box {box} → Area = {area}")

    print("\n" + "=" * 60)
    print("  All checks passed!")
    print("=" * 60)
