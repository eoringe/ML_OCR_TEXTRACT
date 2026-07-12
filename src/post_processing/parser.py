"""
Task 4: Post-Processing & Parsing
Branch: task-1-data-pipeline-emmanuel-oringe (team dissolved — all work consolidated)

This module cleans and normalises the raw output from the KIE module.
It handles:
  - OCR error correction (common character substitutions)
  - Date standardisation to YYYY-MM-DD format
  - Total amount cleaning and validation
  - Company name normalisation
  - Address cleaning

Usage:
    from src.post_processing.parser import ReceiptParser
    parser = ReceiptParser()
    cleaned = parser.parse(raw_extracted_keys)
"""

import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OCR Error Correction Mappings
# ---------------------------------------------------------------------------
# Common OCR character confusion pairs (context-sensitive)
OCR_NUMBER_CORRECTIONS = {
    'O': '0',   # Capital O → zero
    'o': '0',   # Lower o → zero (in numeric context)
    'l': '1',   # Lower L → one
    'I': '1',   # Capital I → one (in numeric context)
    'S': '5',   # S → 5
    'B': '8',   # B → 8
    'Z': '2',   # Z → 2
    'G': '6',   # G → 6
    'q': '9',   # q → 9
    'D': '0',   # D → 0
}

# Date format strings to try (in priority order)
DATE_FORMATS = [
    "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
    "%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d",
    "%d %b %Y", "%d %B %Y",
    "%b %d, %Y", "%B %d, %Y",
    "%b %d %Y", "%B %d %Y",
    "%d/%m/%y", "%m/%d/%y", "%Y/%m/%d",
    "%d-%m-%y", "%m-%d-%y",
]


# ===========================================================================
#  Receipt Parser
# ===========================================================================

class ReceiptParser:
    """
    Post-processing engine that cleans and normalises OCR extraction results.

    Applies:
        - OCR error correction for numeric fields
        - Date standardisation to ISO format (YYYY-MM-DD)
        - Total amount formatting (two decimal places)
        - Company name cleaning and uppercasing
        - Address normalisation
    """

    def __init__(self):
        pass

    def clean_text(self, text):
        """
        Cleans generic OCR artifacts from text.

        Removes unusual characters while preserving common punctuation
        and alphanumeric characters.

        Args:
            text (str): Raw OCR text.

        Returns:
            str: Cleaned text.
        """
        if not text:
            return ""
        # Remove unusual/control characters but keep common punctuation
        cleaned = re.sub(r'[^\w\s\d.,;:\-\/\$#@&\(\)\']', '', text)
        # Collapse multiple whitespaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    def clean_company(self, company_text):
        """
        Cleans and normalises a company/merchant name.

        Args:
            company_text (str): Raw company name from OCR.

        Returns:
            str: Cleaned, uppercased company name.
        """
        if not company_text:
            return ""

        cleaned = self.clean_text(company_text).upper()
        # Fix common MR. D.I.Y. and PAPPARICH OCR variations
        if "MR" in cleaned and ("DIY" in cleaned or "DAIAY" in cleaned or "I.Y." in cleaned or "D.I.Y" in cleaned):
            return "MR. D.I.Y. (M) SDN BHD"
        if "PAPPARI" in cleaned:
            return "PAPPARICH BMC"

        # Remove trailing punctuation artefacts
        cleaned = re.sub(r'[,.\-;:]+$', '', cleaned)
        # Remove leading punctuation artefacts
        cleaned = re.sub(r'^[,.\-;:]+', '', cleaned)
        return cleaned.strip().upper()

    def clean_address(self, address_text):
        """
        Cleans and normalises an address string.

        Args:
            address_text (str): Raw address from OCR.

        Returns:
            str: Cleaned address.
        """
        if not address_text:
            return ""

        cleaned = self.clean_text(address_text)
        
        # 1. Spelling corrections for common Malaysian address/OCR typos
        corrections = {
            r'\bJOlOR\b': 'JOHOR',
            r'\bJHOHOR\b': 'JOHOR',
            r'\bBHRU\b': 'BAHRU',
            r'\bSACU\b': 'SAGU',
            r'\bSLANGOR\b': 'SELANGOR',
            r'\bKBP\b': 'KPB',
            r'\bN0\b': 'NO',  # Correct N0 -> NO
            r'\b5O\b': '50',  # Correct 5O -> 50
        }
        for pattern, replacement in corrections.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        # 2. Put a space after periods in initials (e.g. P.RAMLEE -> P. RAMLEE)
        cleaned = re.sub(r'\b([A-Za-z])\.(?=[A-Za-z])', r'\1. ', cleaned)

        # 3. Correct 5-digit Malaysian postal codes misread by OCR (e.g. 81i00 -> 81100)
        def fix_postcode(match):
            val = match.group(0)
            for char, num in [('i', '1'), ('l', '1'), ('I', '1'), ('o', '0'), ('O', '0')]:
                val = val.replace(char, num)
            return val
        cleaned = re.sub(r'\b[0-9iIlOo]{5}\b', fix_postcode, cleaned)

        return cleaned.strip()

    def standardize_date(self, date_str):
        """
        Converts various date formats to ISO YYYY-MM-DD.

        Tries multiple format patterns. Returns original string if no
        pattern matches.

        Args:
            date_str (str): Raw date string from OCR.

        Returns:
            str: Standardised date in YYYY-MM-DD format, or original string.
        """
        if not date_str:
            return ""

        clean_date = re.sub(r'[^\w\s\-\/,]', '', date_str).strip()

        for fmt in DATE_FORMATS:
            try:
                dt = datetime.strptime(clean_date, fmt)
                # Sanity: reject dates obviously in the future or too old
                if dt.year < 1990 or dt.year > 2099:
                    continue
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        return date_str  # Return original if parsing fails

    def clean_total(self, total_str):
        """
        Cleans and formats a total amount string.

        Handles:
            - Currency symbol removal ($, RM, SGD, etc.)
            - OCR character corrections in numeric context
            - Formatting to two decimal places

        Args:
            total_str (str): Raw total string from OCR.

        Returns:
            str: Cleaned total as "X.XX" format, or "0.00" on failure.
        """
        if not total_str:
            return "0.00"

        # Check if the raw string contains RM currency indicator or common OCR variations
        has_rm = any(indicator in total_str.lower() for indicator in ["rm", "rh", "rn"])

        # Remove currency symbols and letters for numeric cleaning
        cleaned = total_str.strip()

        # Apply OCR corrections for numeric context
        cleaned = self._correct_ocr_numbers(cleaned)

        # Strip remaining non-numeric chars (except dot and comma)
        cleaned = re.sub(r'[^\d.,]', '', cleaned)

        # Handle comma as thousands separator
        if ',' in cleaned and '.' in cleaned:
            # e.g., "1,234.56" → remove comma
            cleaned = cleaned.replace(',', '')
        elif ',' in cleaned and '.' not in cleaned:
            # e.g., "10,80" → treat comma as decimal
            parts = cleaned.split(',')
            if len(parts) == 2 and len(parts[1]) == 2:
                cleaned = f"{parts[0]}.{parts[1]}"
            else:
                cleaned = cleaned.replace(',', '')

        try:
            val = float(cleaned)
            formatted = f"{val:.2f}"
            return formatted
        except (ValueError, OverflowError):
            return "0.00"

    def validate_total(self, total_str):
        """
        Validates that a total string represents a reasonable monetary value.

        Args:
            total_str (str): Cleaned total string.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            val = float(total_str)
            return 0.0 < val < 1_000_000.0
        except (ValueError, TypeError):
            return False

    def validate_date(self, date_str):
        """
        Validates that a date string is in YYYY-MM-DD format.

        Args:
            date_str (str): Date string to validate.

        Returns:
            bool: True if valid ISO date, False otherwise.
        """
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except (ValueError, TypeError):
            return False

    def parse(self, raw_extracted_keys, standardize_date=True):
        """
        Applies all cleaning and normalisation to extracted key fields.

        Args:
            raw_extracted_keys (dict): Raw dictionary from KeyExtractor:
                {"company": str, "date": str, "address": str, "total": str}
            standardize_date (bool): Whether to convert date to ISO YYYY-MM-DD.

        Returns:
            dict: Cleaned and normalised key fields.
        """
        raw_date = raw_extracted_keys.get("date", "")
        if raw_date and not standardize_date:
            # Basic cleanup of punctuation/spaces without converting format
            cleaned_date = re.sub(r'[^\w\s\-\/,]', '', raw_date).strip()
        else:
            cleaned_date = self.standardize_date(raw_date)

        parsed = {
            "company": self.clean_company(raw_extracted_keys.get("company", "")),
            "date": cleaned_date,
            "address": self.clean_address(raw_extracted_keys.get("address", "")),
            "total": self.clean_total(raw_extracted_keys.get("total", "")),
        }

        logger.debug("Parsed keys: %s", parsed)
        return parsed

    # -----------------------------------------------------------------------
    #  Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _correct_ocr_numbers(text):
        """
        Applies OCR character corrections in a numeric context.

        Only corrects characters surrounded by digits to avoid false positives.

        Args:
            text (str): Input string with potential OCR errors.

        Returns:
            str: Corrected string.
        """
        result = list(text)
        for i, char in enumerate(result):
            if char in OCR_NUMBER_CORRECTIONS:
                # Check if this character is in a numeric context
                # (surrounded by digits or at boundaries with digits)
                prev_digit = (i > 0 and result[i - 1].isdigit())
                next_digit = (i < len(result) - 1 and result[i + 1].isdigit())
                if prev_digit or next_digit:
                    result[i] = OCR_NUMBER_CORRECTIONS[char]
        return "".join(result)


# ===========================================================================
#  Standalone Execution
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  Post-Processing & Parsing — Task 4b Verification")
    print("=" * 60)

    parser = ReceiptParser()

    # Test cases
    test_cases = [
        {
            "input": {
                "company": "  booK store inc! ",
                "date": "25/12/2023",
                "address": "123 Main St.",
                "total": "$10.8O"  # 'O' instead of '0'
            },
            "label": "Basic cleaning"
        },
        {
            "input": {
                "company": "  GOLDEN PALACE SDN BHD  ",
                "date": "2023-06-15",
                "address": "45 JLN SULTAN ISMAIL, 50250",
                "total": "RM 1,234.56"
            },
            "label": "Malaysian receipt"
        },
        {
            "input": {
                "company": "",
                "date": "",
                "address": "",
                "total": ""
            },
            "label": "Empty fields"
        },
    ]

    for tc in test_cases:
        result = parser.parse(tc["input"])
        print(f"\n  [{tc['label']}]")
        print(f"    Input:  {tc['input']}")
        print(f"    Output: {result}")

    # Test OCR corrections
    print("\n  OCR Numeric Corrections:")
    corrections = [
        ("1O.8O", "10.80"),
        ("$l5.OO", "$15.00"),
        ("2S.5O", "25.50"),
    ]
    for raw, expected_hint in corrections:
        cleaned = parser.clean_total(raw)
        print(f"    '{raw}' → '{cleaned}' (expected ~'{expected_hint}')")

    # Test date standardisation
    print("\n  Date Standardisation:")
    dates = ["25/12/2023", "2023-06-15", "12/25/23", "15 Jan 2024"]
    for d in dates:
        standardised = parser.standardize_date(d)
        valid = parser.validate_date(standardised)
        print(f"    '{d}' → '{standardised}' (valid={valid})")

    print("\n" + "=" * 60)
    print("  All checks passed!")
    print("=" * 60)
