import os
import sys
import cv2
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_pipeline.prep import split_dataset
from src.recognition.recognize import TextRecognizer
from src.kie.extract import KeyExtractor
from src.post_processing.parser import ReceiptParser
from src.utils.helpers import merge_boxes_into_lines

def debug_address():
    data_dir = "dataset"
    _, _, test_files = split_dataset(data_dir)
    eval_files = test_files[:10]
    
    recognizer = TextRecognizer()
    extractor = KeyExtractor()
    parser = ReceiptParser()
    
    print("Evaluating addresses & totals...")
    for filename in eval_files:
        base_name = os.path.splitext(filename)[0]
        img_path = os.path.join(data_dir, "img", filename)
        gt_path = os.path.join(data_dir, "key", f"{base_name}.json")
        
        if not os.path.exists(gt_path):
            continue
            
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
            
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        results = recognizer.detect_and_recognize(img_rgb)
        boxes = [r[0] for r in results]
        transcripts = [r[1] for r in results]
        
        merged_boxes, merged_transcripts = merge_boxes_into_lines(boxes, transcripts)
        raw_keys = extractor.extract_keys(merged_transcripts, merged_boxes)
        parsed_keys = parser.parse(raw_keys, standardize_date=False)
        
        print(f"\n--- File: {filename} ---")
        print(f"GT Address:   '{gt.get('address', '')}'")
        print(f"Pred Address: '{parsed_keys.get('address', '')}'")
        print(f"GT Total:     '{gt.get('total', '')}'")
        print(f"Pred Total:   '{parsed_keys.get('total', '')}'")

if __name__ == "__main__":
    debug_address()
