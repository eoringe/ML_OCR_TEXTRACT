import re
import json

class KeyExtractor:
    """
    Task 4: Key Information Extraction (KIE).
    Responsible for classifying recognized text snippets into target fields:
    - Company (Merchant Name)
    - Date
    - Address
    - Total
    """
    def __init__(self, model_path=None):
        self.model_path = model_path
        # If using LayoutLM, load tokenizer and model here
        
    def train_epoch(self, train_loader, optimizer):
        """
        Placeholder training loop for deep learning KIE (e.g. LayoutLM sequence labeling).
        """
        pass

    def extract_keys(self, transcripts, boxes=None):
        """
        Extracts key information from receipt text transcripts and bounding boxes.
        
        Args:
            transcripts (list): List of strings representing recognized text.
            boxes (list, optional): List of bounding boxes corresponding to transcripts.
        Returns:
            dict: Structured keys {"company": str, "date": str, "address": str, "total": str}
        """
        # Baseline Implementation: Rule-based Heuristic
        # Highly accurate baseline when OCR is clean, and serves as an ideal baseline for developers.
        result = {
            "company": "",
            "date": "",
            "address": "",
            "total": ""
        }
        
        if not transcripts:
            return result
            
        full_text = "\n".join(transcripts)
        
        # 1. Company Name: Typically the first few lines of the receipt
        candidate_companies = []
        for line in transcripts[:4]:
            clean_line = line.strip()
            # Exclude lines that start with numbers (addresses, tel, dates)
            if clean_line and not re.match(r'^\d', clean_line) and len(clean_line) > 2:
                candidate_companies.append(clean_line)
        if candidate_companies:
            result["company"] = candidate_companies[0]
            
        # 2. Date: Regular expressions to match typical date formats
        date_patterns = [
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',      # DD/MM/YYYY or MM/DD/YYYY
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',      # YYYY/MM/DD
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b' # DD MMM YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            if matches:
                result["date"] = matches[0]
                break
                
        # 3. Total: Match patterns containing "TOTAL", "AMOUNT", "NET", etc.
        total_patterns = [
            r'(?:total|grand\s+total|net\s+total|amount\s+due|total\s+due|sub\s*total)\s*(?::|rs|rm|sgd|usd|\$)?\s*([\d,]+\.\d{2})\b',
            r'\b(?:total|amount|cash|visa|mastercard|net)\b.*?(\d+\.\d{2})\b'
        ]
        
        highest_total = 0.0
        for pattern in total_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            for m in matches:
                try:
                    val = float(m.replace(",", ""))
                    if val > highest_total:
                        highest_total = val
                except ValueError:
                    continue
        if highest_total > 0:
            result["total"] = f"{highest_total:.2f}"
            
        # 4. Address: Match lines that contain street keywords
        street_keywords = ['street', 'str', 'rd', 'road', 'ave', 'avenue', 'jln', 'jalan', 'highway', 'hwy', 'building', 'bldg', 'plaza', 'mall']
        address_lines = []
        for line in transcripts[1:10]:  # Usually follows company name
            clean_line = line.strip()
            # Check if any street keyword or postal code is in the line
            has_keyword = any(keyword in clean_line.lower() for keyword in street_keywords)
            has_postal = re.search(r'\b\d{5,6}\b', clean_line) is not None
            if has_keyword or has_postal:
                address_lines.append(clean_line)
                
        if address_lines:
            result["address"] = ", ".join(address_lines[:3])
        else:
            # Fallback: take lines 2 and 3 if they look like contact info
            fallback = [t.strip() for t in transcripts[1:3] if len(t.strip()) > 5]
            result["address"] = ", ".join(fallback)
            
        return result

if __name__ == "__main__":
    extractor = KeyExtractor()
    sample_ocr = [
        "BOOK STORE INC",
        "123 MAIN STREET, NY 10001",
        "TEL: 555-1234",
        "DATE: 12/25/2023",
        "ITEM A        $10.00",
        "SUBTOTAL       $10.00",
        "TAX             $0.80",
        "TOTAL          $10.80",
        "CASH PAID      $20.00"
    ]
    
    extracted = extractor.extract_keys(sample_ocr)
    print("Extracted Keys:")
    print(json.dumps(extracted, indent=2))
