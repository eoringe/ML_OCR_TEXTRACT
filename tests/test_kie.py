"""
Unit tests for Task 4a: Key Information Extraction (KIE)
Run with: python -m pytest tests/test_kie.py -v
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.kie.extract import KeyExtractor, SROIE_FIELDS


class TestKeyExtractorBasic(unittest.TestCase):
    """Tests for KeyExtractor basic functionality."""

    def setUp(self):
        self.extractor = KeyExtractor()

    def test_empty_transcripts(self):
        result = self.extractor.extract_keys([])
        for field in SROIE_FIELDS:
            self.assertIn(field, result)
            self.assertEqual(result[field], "")

    def test_none_transcripts(self):
        result = self.extractor.extract_keys(None)
        for field in SROIE_FIELDS:
            self.assertEqual(result[field], "")

    def test_returns_all_fields(self):
        transcripts = ["STORE NAME", "123 Main St", "Date: 01/01/2024", "Total $10.00"]
        result = self.extractor.extract_keys(transcripts)
        for field in SROIE_FIELDS:
            self.assertIn(field, result)


class TestCompanyExtraction(unittest.TestCase):
    """Tests for company name extraction."""

    def setUp(self):
        self.extractor = KeyExtractor()

    def test_first_line_company(self):
        transcripts = ["BOOK STORE INC", "123 Main Street", "Total $10.00"]
        result = self.extractor.extract_keys(transcripts)
        self.assertEqual(result["company"], "BOOK STORE INC")

    def test_skips_numeric_first_line(self):
        transcripts = ["12345", "STORE NAME", "123 Main St"]
        result = self.extractor.extract_keys(transcripts)
        self.assertEqual(result["company"], "STORE NAME")

    def test_skips_date_line(self):
        transcripts = ["01/01/2024", "MY STORE", "Some address"]
        result = self.extractor.extract_keys(transcripts)
        self.assertEqual(result["company"], "MY STORE")

    def test_short_lines_skipped(self):
        transcripts = ["AB", "REAL COMPANY NAME", "Address line"]
        result = self.extractor.extract_keys(transcripts)
        self.assertEqual(result["company"], "REAL COMPANY NAME")


class TestDateExtraction(unittest.TestCase):
    """Tests for date extraction."""

    def setUp(self):
        self.extractor = KeyExtractor()

    def test_dd_mm_yyyy_slash(self):
        transcripts = ["STORE", "Date: 25/12/2023"]
        result = self.extractor.extract_keys(transcripts)
        self.assertEqual(result["date"], "25/12/2023")

    def test_yyyy_mm_dd(self):
        transcripts = ["STORE", "2023/06/15 receipt"]
        result = self.extractor.extract_keys(transcripts)
        self.assertEqual(result["date"], "2023/06/15")

    def test_dd_month_yyyy(self):
        transcripts = ["STORE", "15 January 2024"]
        result = self.extractor.extract_keys(transcripts)
        self.assertIn("15", result["date"])
        self.assertIn("2024", result["date"])

    def test_no_date_returns_empty(self):
        transcripts = ["STORE", "No date here"]
        result = self.extractor.extract_keys(transcripts)
        self.assertEqual(result["date"], "")

    def test_dd_mm_yy(self):
        transcripts = ["STORE", "Invoice 25-12-23"]
        result = self.extractor.extract_keys(transcripts)
        self.assertIn("25", result["date"])


class TestTotalExtraction(unittest.TestCase):
    """Tests for total amount extraction."""

    def setUp(self):
        self.extractor = KeyExtractor()

    def test_basic_total(self):
        transcripts = ["STORE", "SUBTOTAL $10.00", "TAX $0.80", "TOTAL $10.80"]
        result = self.extractor.extract_keys(transcripts)
        # Should pick highest total-related value
        self.assertIn(".", result["total"])

    def test_total_with_comma(self):
        transcripts = ["STORE", "GRAND TOTAL 1,234.56"]
        result = self.extractor.extract_keys(transcripts)
        self.assertEqual(result["total"], "1234.56")

    def test_total_with_currency(self):
        transcripts = ["STORE", "TOTAL RM 45.50"]
        result = self.extractor.extract_keys(transcripts)
        self.assertEqual(result["total"], "45.50")

    def test_no_total_returns_empty(self):
        transcripts = ["STORE", "Just some text"]
        result = self.extractor.extract_keys(transcripts)
        self.assertEqual(result["total"], "")

    def test_highest_total_selected(self):
        transcripts = [
            "STORE",
            "SUBTOTAL $10.00",
            "TAX $0.80",
            "TOTAL $10.80",
            "CASH $20.00",
        ]
        result = self.extractor.extract_keys(transcripts)
        total = float(result["total"])
        self.assertGreaterEqual(total, 10.80)


class TestAddressExtraction(unittest.TestCase):
    """Tests for address extraction."""

    def setUp(self):
        self.extractor = KeyExtractor()

    def test_street_keyword(self):
        transcripts = ["STORE", "123 MAIN STREET, NY 10001"]
        result = self.extractor.extract_keys(transcripts)
        self.assertIn("MAIN STREET", result["address"])

    def test_postal_code(self):
        transcripts = ["STORE", "ADDRESS LINE, 50250"]
        result = self.extractor.extract_keys(transcripts)
        self.assertIn("50250", result["address"])

    def test_road_keyword(self):
        transcripts = ["STORE", "45 JALAN SULTAN ISMAIL"]
        result = self.extractor.extract_keys(transcripts)
        self.assertIn("JALAN", result["address"])

    def test_fallback_address(self):
        transcripts = ["STORE", "SOME LONG ADDRESS LINE WITHOUT KEYWORDS"]
        result = self.extractor.extract_keys(transcripts)
        # Fallback should use lines 2-3
        self.assertNotEqual(result["address"], "")

    def test_no_address(self):
        transcripts = ["X"]
        result = self.extractor.extract_keys(transcripts)
        # May return empty or not
        self.assertIsInstance(result["address"], str)


class TestConfidenceScoring(unittest.TestCase):
    """Tests for extract_keys_with_confidence()."""

    def setUp(self):
        self.extractor = KeyExtractor()

    def test_returns_fields_and_confidence(self):
        transcripts = ["STORE", "123 Main Street", "Date: 01/01/2024", "TOTAL $10.80"]
        result = self.extractor.extract_keys_with_confidence(transcripts)
        self.assertIn("fields", result)
        self.assertIn("confidence", result)

    def test_confidence_ranges(self):
        transcripts = ["STORE", "123 Main Street", "Date: 01/01/2024", "TOTAL $10.80"]
        result = self.extractor.extract_keys_with_confidence(transcripts)
        for field in SROIE_FIELDS:
            self.assertGreaterEqual(result["confidence"][field], 0.0)
            self.assertLessEqual(result["confidence"][field], 1.0)

    def test_empty_fields_zero_confidence(self):
        result = self.extractor.extract_keys_with_confidence([])
        for field in SROIE_FIELDS:
            self.assertEqual(result["confidence"][field], 0.0)


class TestExtractFromFullText(unittest.TestCase):
    """Tests for extract_from_full_text()."""

    def setUp(self):
        self.extractor = KeyExtractor()

    def test_full_text_extraction(self):
        text = "BOOK STORE INC\n123 MAIN STREET\nDATE: 12/25/2023\nTOTAL $10.80"
        result = self.extractor.extract_from_full_text(text)
        self.assertEqual(result["company"], "BOOK STORE INC")
        self.assertNotEqual(result["date"], "")

    def test_empty_text(self):
        result = self.extractor.extract_from_full_text("")
        for field in SROIE_FIELDS:
            self.assertEqual(result[field], "")


class TestSpatialAwareness(unittest.TestCase):
    """Tests for spatial-aware extraction with bounding boxes."""

    def setUp(self):
        self.extractor = KeyExtractor()

    def test_with_matching_boxes(self):
        transcripts = ["STORE NAME", "123 Main Street", "TOTAL $10.80"]
        boxes = [
            [10, 10, 200, 10, 200, 30, 10, 30],
            [10, 40, 200, 40, 200, 60, 10, 60],
            [10, 300, 200, 300, 200, 320, 10, 320],
        ]
        result = self.extractor.extract_keys(transcripts, boxes)
        self.assertIsInstance(result, dict)

    def test_mismatched_boxes_ignored(self):
        transcripts = ["STORE NAME", "123 Main Street"]
        boxes = [[10, 10, 200, 10, 200, 30, 10, 30]]  # Only 1 box for 2 transcripts
        # Should fall back to index-based positioning
        result = self.extractor.extract_keys(transcripts, boxes)
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
