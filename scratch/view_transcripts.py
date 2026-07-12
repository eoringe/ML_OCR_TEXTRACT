import os
import sys
import cv2
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.recognition.recognize import TextRecognizer
from src.utils.helpers import merge_boxes_into_lines

def view_transcripts():
    data_dir = "dataset"
    target_files = ["105.jpg", "205.jpg", "034.jpg", "508.jpg", "080.jpg", "389.jpg"]
    
    recognizer = TextRecognizer()
    out_path = "scratch/transcripts_output.txt"
    print(f"Writing transcripts to {out_path}...")
    
    with open(out_path, "w", encoding="utf-8") as out:
        for filename in target_files:
            img_path = os.path.join(data_dir, "img", filename)
            if not os.path.exists(img_path):
                continue
                
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            results = recognizer.detect_and_recognize(img_rgb)
            boxes = [r[0] for r in results]
            transcripts = [r[1] for r in results]
            
            merged_boxes, merged_transcripts = merge_boxes_into_lines(boxes, transcripts)
            
            out.write(f"\n=================== File: {filename} ===================\n")
            for idx, t in enumerate(merged_transcripts):
                out.write(f"Line {idx}: '{t}'\n")
    print("Done!")

if __name__ == "__main__":
    view_transcripts()
