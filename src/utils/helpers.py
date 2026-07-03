import os
import cv2
import numpy as np

def create_directory_structure(root_path):
    """
    Ensures that all key directories for the OCR Textract pipeline are created.
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
        "checkpoints"
    ]
    for d in dirs:
        path = os.path.join(root_path, d)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")

def calculate_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) of two 4-point (x1, y1, x2, y2) rectangles.
    Useful for evaluation or non-maximum suppression (NMS) in detection models.
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Bounding points of intersection
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
    
    if union_area == 0.0:
        return 0.0
        
    return intersection_area / union_area

def draw_polygon_on_image(image, box, label=None, color=(0, 255, 0), thickness=2):
    """
    Draws an 8-point bounding box polygon on an image.
    """
    pts = np.array(box).reshape((-1, 2)).astype(np.int32)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
    if label:
        cv2.putText(
            image, label, (pts[0][0], pts[0][1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1
        )
    return image
