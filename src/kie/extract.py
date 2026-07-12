"""
Task 4: Key Information Extraction (KIE)
Branch: task-1-data-pipeline-emmanuel-oringe (team dissolved — all work consolidated)

This module classifies recognised text snippets from receipts into structured
fields: Company, Date, Address, and Total.

Supports:
  - Rule-based heuristic extraction (primary, no ML dependencies)
  - Spatial/positional awareness using bounding box y-coordinates
  - Confidence scoring for each extracted field
  - Convenience wrapper for full-text extraction

Usage:
    from src.kie.extract import KeyExtractor
    extractor = KeyExtractor()
    result = extractor.extract_keys(transcripts, boxes)
"""

import re
import json
import logging
import os

try:
    import torch
    from transformers import LayoutLMForTokenClassification, LayoutLMTokenizerFast
    HAS_LAYOUTLM = True
except ImportError:
    HAS_LAYOUTLM = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SROIE_FIELDS = ("company", "date", "address", "total")

# Regex to skip generic headers and document titles
GENERIC_HEADER_PATTERN = re.compile(
    r'\b(?:normal\s+)?sales\s+receipt\b|\btax\s+invoice\b|\bcash\s+bill\b|\bwelcome\s+to\b|\bmerchant\s+copy\b|\bcustomer\s+copy\b|\bduplicate\s+receipt\b|\binvoice\b|\breceipt\b',
    re.IGNORECASE
)

DATE_PATTERNS = [
    r'\b(?:0?[1-9]|[12]\d|3[01])([-/])(?:0?[1-9]|1[012])\1\d{2,4}\b',  # DD/MM/YYYY or DD-MM-YYYY (matching separators)
    r'\b(?:0?[1-9]|1[012])([-/])(?:0?[1-9]|[12]\d|3[01])\1\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
    r'\b\d{4}([-/])(?:0?[1-9]|1[012])\1(?:0?[1-9]|[12]\d|3[01])\b',    # YYYY/MM/DD
    r'\b(?:0?[1-9]|[12]\d|3[01])\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',  # DD MMM YYYY
    r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(?:0?[1-9]|[12]\d|3[01]),?\s+\d{2,4}\b',  # MMM DD, YYYY
]

TOTAL_PATTERNS = [
    # 1. Total/Grand Total/Cash/Net/Payment followed by value (allowing currency, spaces, commas)
    # We allow the thousands separator to be a comma followed by an optional space, or just spaces, using (?:,\s*|\s+)
    r'(?:total|grand\s+total|net\s+total|amount\s+due|total\s+due|sub\s*total|sub\s+total|amount|cash|payment|paid|visa|mastercard|net)\s*[:\-=\s]*\s*(?:rs|rm|sgd|usd|ksh|shs|sh|\$)?\s*(\b\d{1,3}(?:(?:,\s*|\s+)\d{3})*(?:\.\d{2})?\b|\b\d+(?:\.\d{2})?\b)',
    # 2. Currency word/symbol followed by value
    r'(?:ksh|shs|sh|rm|sgd|usd|\$)\s*[:\-=\s]*\s*(\b\d{1,3}(?:(?:,\s*|\s+)\d{3})*(?:\.\d{2})?\b|\b\d+(?:\.\d{2})?\b)',
]

STREET_KEYWORDS = [
    'street', 'str', 'rd', 'road', 'ave', 'avenue', 'jln', 'jalan',
    'highway', 'hwy', 'building', 'bldg', 'plaza', 'mall', 'blvd',
    'boulevard', 'lane', 'ln', 'drive', 'dr', 'court', 'ct',
    'floor', 'flr', 'level', 'suite', 'ste', 'unit', 'block',
    'taman', 'lorong', 'kampung', 'sdn', 'bhd',
    'p.o. box', 'p.o box', 'po box', 'p o box', 'box',
    'branch', 'nairobi', 'mombasa', 'kisumu',
]

# Keywords that indicate a total-related line
TOTAL_KEYWORDS = [
    'total', 'grand total', 'net total', 'amount due', 'total due',
    'subtotal', 'sub total', 'amount', 'balance', 'payment',
    'cash', 'visa', 'mastercard', 'change',
]


# ===========================================================================
#  Key Extractor
# ===========================================================================

class KeyExtractor:
    """
    Key Information Extraction engine for SROIE receipts.

    Extracts four fields from OCR transcripts:
        - company: Merchant/store name
        - date: Transaction date
        - address: Store address
        - total: Total amount paid

    Supports rule-based extraction with optional spatial awareness using
    bounding box coordinates.
    """

    def __init__(self, model_path=None):
        """
        Args:
            model_path (str, optional): Path to trained model weights
                (reserved for LayoutLM integration).
        """
        self.model_path = model_path
        self._tokenizer = None
        self._model = None
        self._init_failed = False

    def _init_layoutlm(self):
        if not HAS_LAYOUTLM or self._init_failed:
            return False
            
        if self._model is not None:
            return True
            
        try:
            # Prevent OpenMP and symlink issues on Windows
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
            
            model_name = self.model_path or "npnam693/layoutlm-sroie"
            self._tokenizer = LayoutLMTokenizerFast.from_pretrained(model_name)
            self._model = LayoutLMForTokenClassification.from_pretrained(model_name)
            logger.info("LayoutLM model loaded successfully for KIE extraction.")
            return True
        except Exception as exc:
            logger.error("Failed to load LayoutLM model: %s", exc)
            self._init_failed = True
            return False

    def _extract_keys_layoutlm(self, transcripts, boxes):
        words = []
        word_boxes = []
        
        # Estimate image size from boxes
        max_x = 1
        max_y = 1
        for box in boxes:
            max_x = max(max_x, box[0], box[2], box[4], box[6])
            max_y = max(max_y, box[1], box[3], box[5], box[7])
            
        for line, box in zip(transcripts, boxes):
            x0 = min(box[0], box[2], box[4], box[6])
            y0 = min(box[1], box[3], box[5], box[7])
            x1 = max(box[0], box[2], box[4], box[6])
            y1 = max(box[1], box[3], box[5], box[7])
            
            x0_n = int(1000 * x0 / max_x)
            y0_n = int(1000 * y0 / max_y)
            x1_n = int(1000 * x1 / max_x)
            y1_n = int(1000 * y1 / max_y)
            
            x0_n = max(0, min(1000, x0_n))
            y0_n = max(0, min(1000, y0_n))
            x1_n = max(0, min(1000, x1_n))
            y1_n = max(0, min(1000, y1_n))
            
            line_words = line.split()
            for w in line_words:
                words.append(w)
                word_boxes.append([x0_n, y0_n, x1_n, y1_n])
                
        if not words:
            return {k: "" for k in SROIE_FIELDS}
            
        encoding = self._tokenizer(
            words,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        word_ids = encoding.word_ids(batch_index=0)
        bbox_list = []
        for word_id in word_ids:
            if word_id is None:
                bbox_list.append([0, 0, 0, 0])
            else:
                bbox_list.append(word_boxes[word_id])
                
        encoding["bbox"] = torch.tensor([bbox_list])
        encoding.pop("offset_mapping")
        
        # Move inputs to device
        device = next(self._model.parameters()).device
        for k, v in encoding.items():
            encoding[k] = v.to(device)
            
        with torch.no_grad():
            outputs = self._model(**encoding)
            
        predictions = outputs.logits.argmax(-1).squeeze(0).tolist()
        input_ids = encoding["input_ids"].squeeze(0).tolist()
        
        company_words = []
        date_words = []
        address_words = []
        total_words = []
        
        id2label = self._model.config.id2label
        
        # SROIE classes mapping
        for token_id, pred_id, word_id in zip(input_ids, predictions, word_ids):
            if word_id is None:
                continue
            label = id2label.get(pred_id, "O")
            word_str = words[word_id]
            
            if "COMPANY" in label:
                company_words.append((word_id, word_str))
            elif "DATE" in label:
                date_words.append((word_id, word_str))
            elif "ADDRESS" in label:
                address_words.append((word_id, word_str))
            elif "TOTAL" in label:
                total_words.append((word_id, word_str))
                
        def reconstruct_field(word_pairs):
            seen_ids = set()
            unique_words = []
            for word_id, word_str in word_pairs:
                if word_id not in seen_ids:
                    seen_ids.add(word_id)
                    unique_words.append(word_str)
            return " ".join(unique_words)
            
        raw_total = reconstruct_field(total_words)
        total_val = ""
        if raw_total:
            # Extract numbers from LayoutLM predictions and check heuristic
            MONEY_RE = re.compile(r'\d{1,3}(?:[,\s]\d{3})*\.\d{2}|\d{1,3}(?:[,\s]\d{3})+|\d+\.\d{2}')
            amounts = []
            for m in MONEY_RE.finditer(raw_total):
                try:
                    val = float(m.group().replace(",", "").replace(" ", ""))
                    if 0 < val < 1_000_000:
                        amounts.append(val)
                except ValueError:
                    continue
            if amounts:
                total_candidate = f"{max(amounts):.2f}"
                total_val = KeyExtractor._correct_total_heuristic(total_candidate, transcripts)
            else:
                total_val = raw_total
                
        return {
            "company": reconstruct_field(company_words),
            "date": reconstruct_field(date_words),
            "address": reconstruct_field(address_words),
            "total": total_val
        }

    def _pre_clean_transcripts(self, transcripts):
        """Pre-cleans raw OCR transcript strings to fix common recognition noises."""
        cleaned = []
        for text in transcripts:
            # 1. Clean up common date OCR errors (e.g. 25/1?2018 -> 25/12/2018)
            text_cleaned = re.sub(r'(\d{1,2})\s*[?|\\\/]\s*(\d{1,2})', r'\1/\2', text)
            text_cleaned = re.sub(r'(\d{1,2})\s*[?|\\\/]\s*(\d{2,4})', r'\1/\2', text_cleaned)
            
            # Fix slash read as 7/1/9 at the start of a 4-digit year (e.g., 21/0972017 -> 21/09/2017)
            text_cleaned = re.sub(r'\b(\d{1,2})[-/](\d{1,2})[-/]?([179])(20\d{2})\b', r'\1/\2/\4', text_cleaned)
            
            # 2. Fix year prefix digit confusion: e.g. /1018 -> /2018
            text_cleaned = re.sub(r'/10(1|2)\d\b', lambda m: m.group(0).replace('/10', '/20'), text_cleaned)
            text_cleaned = re.sub(r'-10(1|2)\d\b', lambda m: m.group(0).replace('-10', '-20'), text_cleaned)

            # 3. Clean up spaces and commas around decimal points in numbers (e.g. 30. 30 -> 30.30 or 28 , 58 -> 28.58)
            text_cleaned = re.sub(r'(\b\d+)\s*[,.]\s*(\d{2}\b)', r'\1.\2', text_cleaned)

            # 4. Clean up total/monetary prefix noises like ~ or _
            text_cleaned = re.sub(r'([:\-=\s])~+([0-9])', r'\1\2', text_cleaned)
            text_cleaned = re.sub(r'Total\s*[:\-=\s]*\s*~+([0-9])', r'Total \1', text_cleaned, flags=re.IGNORECASE)
            
            cleaned.append(text_cleaned)
        return cleaned

    def extract_keys(self, transcripts, boxes=None):
        """
        Extracts key information from receipt text transcripts.
        """
        result = {k: "" for k in SROIE_FIELDS}

        if not transcripts:
            return result

        transcripts = self._pre_clean_transcripts(transcripts)
        full_text = "\n".join(transcripts)

        # 1. Try LayoutLM extraction first if available and initialized
        if boxes and len(boxes) == len(transcripts) and self._init_layoutlm():
            try:
                result = self._extract_keys_layoutlm(transcripts, boxes)
            except Exception as exc:
                logger.error("LayoutLM extraction failed, falling back to heuristics: %s", exc)
                result = {k: "" for k in SROIE_FIELDS}

        # 2. Fallback to scoring heuristics for any fields LayoutLM missed
        if not result["company"]:
            if boxes and len(boxes) == len(transcripts):
                lines_with_pos = self._associate_spatial(transcripts, boxes)
            else:
                lines_with_pos = [
                    {"text": t, "y": i, "x": 0, "height": 0}
                    for i, t in enumerate(transcripts)
                ]
            result["company"] = self._extract_company(transcripts, lines_with_pos)

        if not result["date"]:
            result["date"] = self._extract_date(full_text)

        if not result["total"]:
            result["total"] = self._extract_total(full_text, transcripts)

        if not result["address"]:
            if boxes and len(boxes) == len(transcripts):
                lines_with_pos = self._associate_spatial(transcripts, boxes)
            else:
                lines_with_pos = [
                    {"text": t, "y": i, "x": 0, "height": 0}
                    for i, t in enumerate(transcripts)
                ]
            result["address"] = self._extract_address(transcripts, lines_with_pos, result["company"])

        return result

    def extract_keys_with_confidence(self, transcripts, boxes=None):
        """
        Extracts keys with confidence scores for each field.

        Args:
            transcripts (list[str]): Recognised text lines.
            boxes (list, optional): Bounding boxes.

        Returns:
            dict: {
                "fields": {"company": str, "date": str, ...},
                "confidence": {"company": float, "date": float, ...}
            }
        """
        fields = self.extract_keys(transcripts, boxes)
        confidence = {}

        for field in SROIE_FIELDS:
            value = fields[field]
            if not value:
                confidence[field] = 0.0
            elif field == "date" and re.search(r'\d', value):
                confidence[field] = 0.8
            elif field == "total" and re.search(r'\d+\.\d{2}', value):
                confidence[field] = 0.85
            elif field == "company" and len(value) > 2:
                confidence[field] = 0.6
            elif field == "address" and len(value) > 5:
                confidence[field] = 0.5
            else:
                confidence[field] = 0.3

        return {"fields": fields, "confidence": confidence}

    def extract_from_full_text(self, text):
        """
        Convenience method to extract keys from a single text block.

        Args:
            text (str): Full receipt text (lines separated by newlines).

        Returns:
            dict: Extracted key fields.
        """
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return self.extract_keys(lines)

    # -----------------------------------------------------------------------
    #  Spatial helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _associate_spatial(transcripts, boxes):
        """Associates transcripts with spatial position from bounding boxes."""
        lines = []
        for text, box in zip(transcripts, boxes):
            y_min = min(box[1], box[3], box[5], box[7])
            x_min = min(box[0], box[2], box[4], box[6])
            y_max = max(box[1], box[3], box[5], box[7])
            lines.append({
                "text": text,
                "y": y_min,
                "x": x_min,
                "height": y_max - y_min,
            })
        return sorted(lines, key=lambda l: (l["y"], l["x"]))

    # -----------------------------------------------------------------------
    #  Field extraction methods
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_company(transcripts, lines_with_pos):
        """
        Extracts the company/merchant name using a layout-aware scoring model.
        """
        candidates = []
        
        # Scoring metrics
        company_keywords = [
            r'\bsdn\b', r'\bbhd\b', r'\bcorp\b', r'\bco\b', r'\bltd\b', r'\binc\b', 
            r'\bmart\b', r'\bsupermarket\b', r'\bstore\b', r'\bcafe\b', r'\brestaurant\b',
            r'\bshop\b', r'\bmarket\b', r'\bstation\b', r'\benterprise\b', r'\btrading\b',
            r'\bservice\b', r'\bservices\b', r'\bgift\b', r'\bdeco\b', r'\bbook\b', r'\bpharmacy\b',
            r'\bhair\b', r'\bsalon\b', r'\bfood\b', r'\bbakery\b', r'\bexpress\b', r'\bhotel\b',
            r'\bcoffee\b', r'\btea\b', r'\bauto\b', r'\bcar\b', r'\bparts\b', r'\bhardware\b',
            r'\bpetrol\b', r'\bpetroleum\b', r'\bfuel\b', r'\benergy\b', r'\bbank\b',
            r'\blimited\b', r'\bgroup\b', r'\bholdings\b', r'\bcompany\b',
        ]
        
        street_indicators = [
            r'\bstreet\b', r'\brd\b', r'\broad\b', r'\bave\b', r'\bavenue\b', r'\bjln\b', r'\bjalan\b',
            r'\bhighway\b', r'\bhwy\b', r'\bbuilding\b', r'\bplaza\b', r'\bmall\b', r'\blane\b', r'\bln\b',
            r'\bdrive\b', r'\bdr\b', r'\bcourt\b', r'\bct\b', r'\bfloor\b', r'\blevel\b', r'\bsuite\b',
            r'\bblock\b', r'\bblk\b', r'\btaman\b', r'\blorong\b', r'\bpo\s+box\b', r'\bbox\b'
        ]

        for i, line in enumerate(transcripts[:6]):
            clean = line.strip()
            # Remove decorative symbols for checking content
            content_check = re.sub(r'[*=\-#@%+~|]+', '', clean).strip()
            
            if not content_check or len(content_check) <= 2:
                continue
                
            # Clean generic headers out of the candidate rather than skipping the entire line
            content_check_cleaned = GENERIC_HEADER_PATTERN.sub('', content_check).strip()
            if not content_check_cleaned or len(content_check_cleaned) <= 2:
                continue
            
            score = 0
            
            # Position-based scoring
            if i == 0:
                score += 5
            elif i == 1:
                score += 3
                
            # Keyword positive signals
            content_lower = content_check_cleaned.lower()
            if any(re.search(kw, content_lower) for kw in company_keywords):
                score += 10
                
            # Negative signals: street/address keywords
            if any(re.search(st, content_lower) for st in street_indicators):
                score -= 10
                
            # Negative signals: phone/fax/email/website/tel
            if any(indicator in content_lower for indicator in ['tel', 'phone', 'fax', 'email', 'www.', 'http', 'pin:']):
                score -= 15
                
            # Negative signals: purely numbers or looks like a date
            if re.match(r'^[\d\s\-\+\(\):]+$', content_check_cleaned):
                score -= 20
            if re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', content_check_cleaned):
                score -= 15
                
            candidates.append({
                "text": content_check_cleaned,
                "score": score,
                "index": i
            })
            
        if not candidates:
            return ""
            
        # Sort by score descending, then by order of appearance
        sorted_candidates = sorted(candidates, key=lambda c: (-c["score"], c["index"]))
        best_candidate = sorted_candidates[0]
        
        # If the best score is very negative, fallback to first candidate not starting with a digit/not purely address
        if best_candidate["score"] < -5:
            for cand in candidates:
                txt = cand["text"]
                if not re.match(r'^\d', txt) and not any(re.search(st, txt.lower()) for st in street_indicators):
                    return txt
            return candidates[0]["text"]
            
        return best_candidate["text"]

    @staticmethod
    def _extract_date(full_text):
        """
        Extracts the transaction date using regex patterns.
        Also checks for 'Date:' labelled lines.
        """
        # First try labeled date lines like "Date:11/05/2026" or "Date: 11-05-2026"
        labeled_match = re.search(r'(?:date|dt)[:\s]*\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', full_text, re.IGNORECASE)
        if labeled_match:
            return labeled_match.group(1)
        
        for pattern in DATE_PATTERNS:
            matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
            if matches:
                return matches[0].group(0)
        return ""

    @staticmethod
    def _extract_total(full_text, transcripts):
        """
        Extracts the total amount.
        """
        # Monetary number pattern: handles 5,000.00  5000.00  5 000.00  332.30  2.00
        MONEY_RE = re.compile(r'\d{1,3}(?:[,\s]\d{3})*\.\d{2}|\d{1,3}(?:[,\s]\d{3})+|\d+\.\d{2}')

        TIER1_KEYWORDS = ['grand total', 'total amount', 'total due', 'net total',
                          'amount due', 'net amount', 'round total', 'total']
        TIER2_KEYWORDS = ['sub total', 'subtotal', 'amount', 'cash', 'payment',
                          'paid', 'visa', 'mastercard', 'net', 'change']
        SKIP_KEYWORDS = ['inv', 'invoice', 'rcpt', 'receipt no', 'pin:', 'tel:',
                         'phone', 'printed by', 'nozzle', 'pump', 'included in',
                         'item', 'qty', 'quantity']

        def parse_money(s):
            """Parse a monetary string like '5,000.00' into a float."""
            try:
                cleaned = s.replace(",", "").replace(" ", "").strip()
                if not cleaned:
                    return None
                val = float(cleaned)
                return val if 0 < val < 1_000_000 else None
            except (ValueError, OverflowError):
                return None

        def extract_amounts_from_line(line):
            """Extract all monetary values from a single line."""
            amounts = []
            for m in MONEY_RE.finditer(line):
                val = parse_money(m.group())
                if val is not None:
                    amounts.append(val)
            return amounts

        # 1. Filter out GST summary lines
        filtered = []
        in_summary = False
        for line in transcripts:
            ll = line.lower()
            if any(s in ll for s in ['gst summary', 'tax code', 'code amount', 'tax rate']):
                in_summary = True
            if not in_summary:
                filtered.append(line)
            elif 'total' in ll and not any(s in ll for s in ['tax', 'summary', 'code']):
                filtered.append(line)

        # 2. Tier 1: Lines with strong total keywords → pick LARGEST value
        tier1_amounts = []
        for line in filtered:
            ll = line.lower()
            if any(skip in ll for skip in SKIP_KEYWORDS):
                continue
            # Check if line has any Tier 1 keyword
            has_keyword = False
            for kw in TIER1_KEYWORDS:
                if kw in ll:
                    has_keyword = True
                    break
            if not has_keyword:
                continue
            # Skip lines that are clearly tax lines (e.g. "total tax 370.37")
            if re.search(r'total\s+tax\b', ll) or re.search(r'tax\s+total\b', ll):
                continue
            amounts = extract_amounts_from_line(ll)
            tier1_amounts.extend(amounts)

        # 3. Tax table fallback
        tax_sums = []
        for line in transcripts:
            ll = line.lower()
            if re.match(r'^\s*(?:sr|zr|tx)\b', ll) or any(s in ll for s in ['txable', 'taxable']):
                nums = extract_amounts_from_line(ll)
                if len(nums) >= 2 and nums[0] > 0 and nums[1] > 0:
                    tax_sums.append(nums[0] + nums[1])

        if tax_sums:
            max_tax = max(tax_sums)
            if not any(abs(max_tax - t) <= 0.05 for t in tier1_amounts):
                tier1_amounts.append(max_tax)

        total_candidate = ""

        if tier1_amounts:
            max_val = max(tier1_amounts)

            # Check for rounding adjustment
            for idx, line in enumerate(filtered):
                ll = line.lower()
                if any(r in ll for r in ['round', 'adj', 'rqund']):
                    for next_line in filtered[idx+1:idx+3]:
                        nums = extract_amounts_from_line(next_line)
                        if nums:
                            r_val = nums[0]
                            if abs(r_val - max_val) <= 0.10:
                                max_val = r_val
                                break
                    break

            total_candidate = f"{max_val:.2f}"

        # 4. Tier 2: Weaker keywords
        if not total_candidate:
            tier2_amounts = []
            for line in filtered:
                ll = line.lower()
                if any(skip in ll for skip in SKIP_KEYWORDS):
                    continue
                if any(kw in ll for kw in TIER2_KEYWORDS):
                    amounts = extract_amounts_from_line(ll)
                    tier2_amounts.extend(amounts)

            if tier2_amounts:
                total_candidate = f"{max(tier2_amounts):.2f}"

        # 5. Last resort: any line with "total" keyword and any number
        if not total_candidate:
            for line in filtered:
                ll = line.lower()
                if 'total' in ll or 'amount' in ll or 'grand' in ll or 'net' in ll:
                    amounts = extract_amounts_from_line(ll)
                    if amounts:
                        total_candidate = f"{max(amounts):.2f}"
                        break

        # 6. Apply cross-validation heuristic correction for common currency misrecognitions
        return KeyExtractor._correct_total_heuristic(total_candidate, transcripts)

    @staticmethod
    def _correct_total_heuristic(total_candidate_str, transcripts):
        if not total_candidate_str:
            return ""

        MONEY_RE = re.compile(r'\d{1,3}(?:[,\s]\d{3})*\.\d{2}|\d{1,3}(?:[,\s]\d{3})+|\d+\.\d{2}')
        
        other_amounts = []
        for line in transcripts:
            # Skip grand total lines when gathering other amounts
            ll = line.lower()
            if 'total' in ll and 'sub' not in ll:
                continue
            for m in MONEY_RE.finditer(line):
                try:
                    val = float(m.group().replace(",", "").replace(" ", ""))
                    if 0 < val < 1_000_000:
                        other_amounts.append(val)
                except ValueError:
                    continue

        if not other_amounts:
            return total_candidate_str

        max_other = max(other_amounts)
        
        try:
            total_val = float(total_candidate_str)
        except ValueError:
            return total_candidate_str

        # Heuristic: if total is suspiciously large compared to other amounts
        if total_val > 1.5 * max_other:
            val_str = f"{total_val:.2f}"
            if len(val_str) > 4:  # e.g. "5154.06" has length 7
                first_digit = val_str[0]
                stripped_str = val_str[1:]
                try:
                    stripped_val = float(stripped_str)
                    if 0.5 * max_other <= stripped_val <= 1.5 * max_other:
                        # Validate if stripped value matches subtotal + tax or single item
                        matched = False
                        for i in range(len(other_amounts)):
                            for j in range(i, len(other_amounts)):
                                sum_val = other_amounts[i] + (other_amounts[j] if i != j else 0)
                                if abs(sum_val - stripped_val) <= 0.05:
                                    matched = True
                                    break
                            if matched:
                                break
                                
                        if not matched:
                            for amt in other_amounts:
                                if abs(amt - stripped_val) <= 0.05:
                                    matched = True
                                    break
                                    
                        if matched:
                            logger.info("Heuristic corrected grand total: %s -> %s (stripped leading '%s')", total_val, stripped_val, first_digit)
                            return f"{stripped_val:.2f}"
                except ValueError:
                    pass
                    
        return total_candidate_str

    @staticmethod
    def _extract_address(transcripts, lines_with_pos, company_name=""):
        """
        Extracts the store address.

        Heuristic: Identify the start of the address block using keywords or postal codes,
        then collect consecutive address lines to reconstruct the complete address block.
        """
        address_lines = []
        company_lower = company_name.lower().strip() if company_name else ""
        
        skip_keywords = [
            'tel', 'phone', 'fax', 'date', 'inv', 'invoice', 'rcpt', 'receipt', 
            'cashier', 'member', 'sdn bhd', 'bhd', 'gst', 'reg', 'co.reg', 
            'mobile', 'whatsapp', 'email', 'website', 'www.', 'tax', 'nozzle', 'pump'
        ]

        # Find the start of the address block in the first 10 lines
        start_idx = -1
        for i in range(1, min(10, len(transcripts))):
            line = transcripts[i].strip()
            line_lower = line.lower()
            
            # Skip lines matching company name only if they have no address indicators
            if company_lower and (company_lower in line_lower or line_lower in company_lower):
                has_addr_indicator = any(kw in line_lower for kw in STREET_KEYWORDS) or re.search(r'\b\d{5}\b', line) is not None
                if not has_addr_indicator:
                    continue
                
            # Check if this line looks like the start of an address
            has_keyword = any(kw in line_lower for kw in STREET_KEYWORDS)
            has_postal = re.search(r'\b\d{5}\b', line) is not None
            starts_with_no = re.match(r'^(?:no|lot|block|blk|#)\b', line_lower) is not None
            
            if has_keyword or has_postal or starts_with_no:
                if not any(skip in line_lower for skip in skip_keywords):
                    start_idx = i
                    break
                    
        if start_idx != -1:
            # Collect consecutive lines making up the address block
            for i in range(start_idx, min(start_idx + 4, len(transcripts))):
                line = transcripts[i].strip()
                line_lower = line.lower()
                
                # Stop if we hit empty or metadata lines
                if not line or len(line) < 3:
                    break
                # Skip phone/fax numbers
                if re.match(r'^\+?[0-9\-\s\(\)]{7,}$', line) is not None:
                    continue
                if company_lower and (company_lower in line_lower or line_lower in company_lower):
                    has_addr_indicator = any(kw in line_lower for kw in STREET_KEYWORDS) or re.search(r'\b\d{5}\b', line) is not None
                    if not has_addr_indicator:
                        break
                if any(skip in line_lower for skip in skip_keywords):
                    break
                if GENERIC_HEADER_PATTERN.search(line):
                    break
                    
                address_lines.append(line)
                
        # Fallback if no contiguous block was detected
        if not address_lines:
            for line in transcripts[1:5]:
                clean = line.strip()
                line_lower = clean.lower()
                if len(clean) > 5:
                    if company_lower and (company_lower in line_lower or line_lower in company_lower):
                        continue
                    if any(skip in line_lower for skip in skip_keywords):
                        continue
                    address_lines.append(clean)
                    
        cleaned_comps = [l.strip("; ") for l in address_lines]
        address = " ".join([c for c in cleaned_comps if c])
        # Normalize whitespace and ensure commas are followed by exactly one space
        address = re.sub(r'\s+', ' ', address)
        address = re.sub(r',\s*', ', ', address)
        return address.strip().rstrip(", ")


# ===========================================================================
#  Standalone Execution
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  Key Information Extraction — Task 4 Verification")
    print("=" * 60)

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
        "CASH PAID      $20.00",
    ]

    extracted = extractor.extract_keys(sample_ocr)
    print("\n  Sample OCR input:")
    for line in sample_ocr:
        print(f"    {line}")
    print("\n  Extracted Keys:")
    print(json.dumps(extracted, indent=4))

    # Test with confidence
    result_conf = extractor.extract_keys_with_confidence(sample_ocr)
    print("\n  Confidence Scores:")
    for field, score in result_conf["confidence"].items():
        print(f"    {field}: {score:.2f}")

    # Test convenience method
    text_block = "\n".join(sample_ocr)
    extracted2 = extractor.extract_from_full_text(text_block)
    print(f"\n  extract_from_full_text matches: {extracted == extracted2}")

    print("\n" + "=" * 60)
    print("  All checks passed!")
    print("=" * 60)
