import re
from datetime import datetime

class ReceiptParser:
    """
    Task 5: Post-Processing & Parsing.
    Cleans up recognized OCR text, corrects spellings, standardizes fields, and formats final JSON.
    """
    def __init__(self):
        pass

    def clean_text(self, text):
        """
        Cleans generic OCR artifacts, replaces common character mistakes.
        """
        if not text:
            return ""
        # Remove unusual characters
        cleaned = re.sub(r'[^\w\s\d.,;:\-\/\$#@&\(\)]', '', text)
        # Fix common OCR confusions (e.g., 'l' instead of '1' in numbers, 'O' instead of '0')
        # This is context specific and usually applied selectively
        return cleaned.strip()

    def standardize_date(self, date_str):
        """
        Converts various date formats (e.g. 12/25/2023, 2023-12-25, 25-Dec-2023) into YYYY-MM-DD.
        """
        if not date_str:
            return ""
            
        clean_date = re.sub(r'[^\w\s\-\/]', '', date_str).strip()
        
        # Try matching DD/MM/YYYY or MM/DD/YYYY
        formats = [
            "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
            "%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d",
            "%d %b %Y", "%d %B %Y",
            "%d/%m/%y", "%m/%d/%y", "%y/%m/%d"
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(clean_date, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
                
        return date_str # Return original if parsing fails

    def clean_total(self, total_str):
        """
        Ensures total is formatted as a clean decimal float string (e.g., "10.80").
        """
        if not total_str:
            return "0.00"
            
        # Strip currency symbols and letters
        clean_val = re.sub(r'[^\d\.]', '', total_str)
        try:
            val = float(clean_val)
            return f"{val:.2f}"
        except ValueError:
            return "0.00"

    def parse(self, raw_extracted_keys):
        """
        Applies a sequence of cleaning and normalization steps to extracted keys.
        
        Args:
            raw_extracted_keys (dict): The dictionary returned by KeyExtractor.
        Returns:
            dict: The finalized, cleaned and validated dictionary.
        """
        parsed_result = {
            "company": self.clean_text(raw_extracted_keys.get("company", "")).upper(),
            "date": self.standardize_date(raw_extracted_keys.get("date", "")),
            "address": self.clean_text(raw_extracted_keys.get("address", "")),
            "total": self.clean_total(raw_extracted_keys.get("total", ""))
        }
        return parsed_result

if __name__ == "__main__":
    parser = ReceiptParser()
    raw = {
        "company": "  booK store inc! ",
        "date": "25/12/23",
        "address": "123 Main St.",
        "total": "$10.8O" # Confused 'O' and '0'
    }
    
    clean = parser.parse(raw)
    print("Parsed output:")
    print(clean)
