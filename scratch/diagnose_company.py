import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.kie.extract import KeyExtractor
from src.post_processing.parser import ReceiptParser

def test_diagnose():
    extractor = KeyExtractor()
    parser = ReceiptParser()
    
    ocr_transcripts = [
        "East Repair Inc. RECEIPT",
        "1912 Harvest Lane",
        "New York, NY 12210",
        "Bill To",
        "Ship To",
        "Receipt # US-001",
        "Date 11/02/2019",
        "Front and rear brake cables 100.00",
        "New set of pedal arms 30.00",
        "Labor 3hrs 15.00",
        "Subtotal 145.00",
        "Sales Tax 6.25% 9.06",
        "TOTAL $5154.06"  # Simulate $ read as 5 (TOTAL 5154.06)
    ]
    
    print("--- Running KeyExtractor.extract_keys ---")
    raw_keys = extractor.extract_keys(ocr_transcripts)
    print("Raw extracted keys:")
    print(raw_keys)
    
    print("\n--- Running ReceiptParser.parse ---")
    parsed = parser.parse(raw_keys)
    print("Parsed keys:")
    print(parsed)

    # Let's verify expectations
    assert parsed["company"] == "EAST REPAIR INC", f"Expected EAST REPAIR INC, got {parsed['company']}"
    assert parsed["total"] == "154.06", f"Expected 154.06, got {parsed['total']}"
    print("\nSUCCESS: All diagnostic tests passed!")

if __name__ == "__main__":
    test_diagnose()
