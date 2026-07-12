"""
SROIE Pipeline Evaluation Script
Runs the text detection, recognition, KIE, and post-processing modules on the
test split of the dataset and prints a comprehensive evaluation report.
"""

import os
import sys
import logging
import cv2

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_pipeline.prep import split_dataset
from src.detection.detect import TextDetector
from src.recognition.recognize import TextRecognizer
from src.kie.extract import KeyExtractor
from src.post_processing.parser import ReceiptParser
from src.utils.evaluate import evaluate_pipeline, generate_report

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def main():
    print("=" * 60)
    print("         SROIE Receipt OCR Pipeline Evaluation")
    print("=" * 60)
    
    data_dir = "dataset"
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found.")
        sys.exit(1)
        
    print("Loading test dataset split...")
    try:
        # split_dataset defaults to 80/10/10 split
        _, _, test_files = split_dataset(data_dir)
        print(f"Discovered {len(test_files)} images in the test set split.")
    except Exception as exc:
        print(f"Error loading dataset: {exc}")
        sys.exit(1)
        
    if not test_files:
        print("No test files found. Make sure the dataset directory is populated.")
        sys.exit(1)
        
    # Limit to first 10 files for a quick evaluation, or run on more if desired
    eval_files = test_files[:10]
    print(f"Running pipeline on first {len(eval_files)} test receipt images...")
    
    # Initialize pipeline components
    print("Initializing pipeline components...")
    detector = TextDetector(method="contour")
    recognizer = TextRecognizer()
    extractor = KeyExtractor()
    parser = ReceiptParser()
    
    # Define pipeline function
    def run_pipeline(image_path):
        # 1. Load image
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. End-to-end pass if using EasyOCR
        if recognizer.backend == "easyocr":
            results = recognizer.detect_and_recognize(img_rgb)
            boxes = [r[0] for r in results]
            transcripts = [r[1] for r in results]
        else:
            boxes = detector.detect(img_rgb)
            transcripts = recognizer.recognize(img_rgb, boxes)
            
        # 3. Merge boxes into lines
        from src.utils.helpers import merge_boxes_into_lines
        merged_boxes, merged_transcripts = merge_boxes_into_lines(boxes, transcripts)
        
        # 4. Key Information Extraction (KIE)
        raw_keys = extractor.extract_keys(merged_transcripts, merged_boxes)
        
        # 5. Post-Processing & Standardisation (Disable date standardization for matching SROIE raw formats)
        parsed_keys = parser.parse(raw_keys, standardize_date=False)
        return parsed_keys

    # Run evaluation
    print("Processing images and calculating metrics...")
    results = evaluate_pipeline(data_dir, eval_files, run_pipeline)
    
    # Print the report
    report = generate_report(results)
    print("\n" + report)

if __name__ == "__main__":
    main()
