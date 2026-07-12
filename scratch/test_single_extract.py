import os
import sys
import cv2
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.recognition.recognize import TextRecognizer
from src.kie.extract import KeyExtractor
from src.utils.helpers import merge_boxes_into_lines

def test_single():
    recognizer = TextRecognizer()
    extractor = KeyExtractor()
    
    files = ["001.jpg", "582.jpg", "205.jpg"]
    
    for filename in files:
        img_path = os.path.join("dataset", "img", filename)
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        results = recognizer.detect_and_recognize(img_rgb)
        boxes = [r[0] for r in results]
        transcripts = [r[1] for r in results]
        
        merged_boxes, merged_transcripts = merge_boxes_into_lines(boxes, transcripts)
        cleaned = extractor._pre_clean_transcripts(merged_transcripts)
        
        print(f"\n=================== File: {filename} ===================")
        for idx, t in enumerate(cleaned):
            print(f"Line {idx}: '{t}'")
            
        # Run manual extractor logic
        # 1. Tier 1 matches
        TIER1_PATTERNS = [
            r'(?:grand\s*total|total\s*amount|total\s*due|round\s*total|net\s*total|amount\s*due|net\s*amount|total)\b.*?(?:rs|rm|sgd|usd|ksh|shs|sh|\$)?\s*~*(?<!@)(?<!@\s)(\b\d{1,3}(?:(?:,\s*|\s+)\d{3})*(?:\.\d{2})?\b|\b\d+(?:\.\d{2})?\b)(?!\s*%)'
        ]
        
        filtered_transcripts = []
        in_summary = False
        for line in cleaned:
            line_lower = line.lower()
            if any(s in line_lower for s in ['gst summary', 'summary', 'tax code', 'code amount', 'tax rate']):
                in_summary = True
            if not in_summary:
                filtered_transcripts.append(line)
            else:
                if 'total' in line_lower and not any(s in line_lower for s in ['tax', 'summary', 'code']):
                    filtered_transcripts.append(line)
                    
        tier1_matches = []
        for line in filtered_transcripts:
            line_lower = line.lower()
            if any(skip in line_lower for skip in ['inv', 'invoice', 'rcpt', 'receipt no', 'pin:', 'tel:', 'phone', 'printed by', 'nozzle', 'pump', 'included in']):
                continue
            for pattern in TIER1_PATTERNS:
                matches = re.findall(pattern, line_lower)
                for m in matches:
                    try:
                        val = float(m.replace(",", "").replace(" ", ""))
                        tier1_matches.append(val)
                    except ValueError:
                        continue
                        
        print(f"Tier 1 matches: {tier1_matches}")
        
        # 2. Tax sums
        tax_sums = []
        for line in cleaned:
            line_lower = line.lower()
            if re.match(r'^\s*(?:sr|zr|tx)\b', line_lower) or any(s in line_lower for s in ['txable', 'taxable', 'gst summary']):
                nums = []
                for x in re.findall(r'\b\d+\.\d{2}\b', line_lower):
                    try:
                        nums.append(float(x))
                    except ValueError:
                        continue
                if len(nums) >= 2 and nums[0] > 0 and nums[1] > 0:
                    tax_sums.append(nums[0] + nums[1])
        print(f"Tax sums: {tax_sums}")
        
        extracted = extractor._extract_total("\n".join(cleaned), cleaned)
        print(f"Extracted result from extract_total: '{extracted}'")

if __name__ == "__main__":
    test_single()
