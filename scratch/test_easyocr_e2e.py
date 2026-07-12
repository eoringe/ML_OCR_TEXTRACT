import os
import sys
import cv2
import json
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detection.detect import TextDetector
from src.recognition.recognize import TextRecognizer
from src.kie.extract import KeyExtractor
from src.post_processing.parser import ReceiptParser
import easyocr

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

def run_benchmark():
    img_path = "dataset/img/000.jpg"
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print("--- Method 1: Two-Stage Pipeline (Contour + EasyOCR recognize) ---")
    detector = TextDetector(method="contour")
    recognizer = TextRecognizer()
    
    t0 = time.time()
    boxes = detector.detect(img_rgb)
    t1 = time.time()
    transcripts = recognizer.recognize(img_rgb, boxes)
    t2 = time.time()
    
    print(f"Detection took: {t1-t0:.4f}s")
    print(f"Recognition took: {t2-t1:.4f}s")
    print(f"Total Two-Stage: {t2-t0:.4f}s")
    print(f"Number of boxes: {len(boxes)}")
    
    print("\n--- Method 2: Native EasyOCR readtext End-to-End ---")
    reader = easyocr.Reader(['en'], gpu=False)
    t3 = time.time()
    # Warmup
    e2e_results = reader.readtext(img_rgb)
    t4 = time.time()
    print(f"Warmup readtext took: {t4-t3:.4f}s")
    
    t5 = time.time()
    e2e_results = reader.readtext(img_rgb)
    t6 = time.time()
    print(f"Second readtext took: {t6-t5:.4f}s")
    print(f"Number of regions: {len(e2e_results)}")
    
    # Let's inspect some of the native results and see if they are cleaner
    native_boxes = []
    native_texts = []
    for res in e2e_results:
        # res format: (box, text, confidence)
        # box: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        b = res[0]
        box_8pt = [b[0][0], b[0][1], b[1][0], b[1][1], b[2][0], b[2][1], b[3][0], b[3][1]]
        native_boxes.append(box_8pt)
        native_texts.append(res[1])
        
    print("\nMerging native boxes into lines...")
    m_boxes, m_texts = merge_boxes_into_lines(native_boxes, native_texts)
    print("Merged Native Transcripts:")
    for i, t in enumerate(m_texts[:15]):
        print(f"  Line {i}: '{t}'")
        
    extractor = KeyExtractor()
    parser = ReceiptParser()
    raw_keys = extractor.extract_keys(m_texts, m_boxes)
    parsed_keys = parser.parse(raw_keys)
    print("\nParsed keys from Native EasyOCR:", json.dumps(parsed_keys, indent=2))

if __name__ == "__main__":
    run_benchmark()
