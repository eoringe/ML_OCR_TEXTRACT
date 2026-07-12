import os
import sys
import cv2
import json

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detection.detect import TextDetector
from src.recognition.recognize import TextRecognizer
from src.kie.extract import KeyExtractor
from src.post_processing.parser import ReceiptParser

def merge_boxes_into_lines(boxes, transcripts, overlap_threshold=0.5):
    if not boxes:
        return [], []
    
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

def test_single():
    img_path = "dataset/img/000.jpg"
    key_path = "dataset/key/000.json"
    
    print(f"Image path: {img_path}")
    print(f"Key path: {key_path}")
    
    if not os.path.exists(img_path):
        print("Image does not exist!")
        return
        
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print("Initializing components...")
    detector = TextDetector(method="contour")
    recognizer = TextRecognizer()
    extractor = KeyExtractor()
    parser = ReceiptParser()
    
    print("Running detection...")
    boxes = detector.detect(img_rgb)
    print(f"Found {len(boxes)} boxes.")
    
    print("Running recognition...")
    transcripts = recognizer.recognize(img_rgb, boxes)
    
    print("Merging boxes into lines...")
    merged_boxes, merged_transcripts = merge_boxes_into_lines(boxes, transcripts)
    print("Merged Transcripts:")
    for i, (b, t) in enumerate(zip(merged_boxes, merged_transcripts)):
        print(f"  Line {i}: '{t}'")
        
    print("Running extraction on merged lines...")
    raw_keys = extractor.extract_keys(merged_transcripts, merged_boxes)
    print("Raw extracted keys:", json.dumps(raw_keys, indent=2))
    
    print("Running parser...")
    parsed_keys = parser.parse(raw_keys)
    print("Parsed keys:", json.dumps(parsed_keys, indent=2))
    
    if os.path.exists(key_path):
        with open(key_path, "r") as f:
            gt = json.load(f)
        print("Ground truth:", json.dumps(gt, indent=2))

if __name__ == "__main__":
    test_single()
