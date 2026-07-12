"""
Unit tests for Task 4b: Post-Processing & Parsing
Run with: python -m pytest tests/test_post_processing.py -v
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.post_processing.parser import ReceiptParser


class TestCleanText(unittest.TestCase):
    """Tests for ReceiptParser.clean_text()."""

    def setUp(self):
        self.parser = ReceiptParser()

    def test_removes_special_chars(self):
        result = self.parser.clean_text("Hello™ World® ©2024")
        self.assertNotIn("™", result)
        self.assertNotIn("®", result)

    def test_preserves_normal_punctuation(self):
        result = self.parser.clean_text("$10.50, item #3")
        self.assertIn("$", result)
        self.assertIn(",", result)
        self.assertIn("#", result)

    def test_collapses_whitespace(self):
        result = self.parser.clean_text("  too   many   spaces  ")
        self.assertEqual(result, "too many spaces")

    def test_empty_string(self):
        self.assertEqual(self.parser.clean_text(""), "")

    def test_none_input(self):
        self.assertEqual(self.parser.clean_text(None), "")


class TestCleanCompany(unittest.TestCase):
    """Tests for ReceiptParser.clean_company()."""

    def setUp(self):
        self.parser = ReceiptParser()

    def test_uppercases(self):
        result = self.parser.clean_company("book store inc")
        self.assertEqual(result, "BOOK STORE INC")

    def test_strips_whitespace(self):
        result = self.parser.clean_company("  ACME CORP  ")
        self.assertEqual(result, "ACME CORP")

    def test_removes_trailing_punctuation(self):
        result = self.parser.clean_company("STORE NAME,.")
        self.assertEqual(result, "STORE NAME")

    def test_removes_leading_punctuation(self):
        result = self.parser.clean_company(",.STORE NAME")
        self.assertEqual(result, "STORE NAME")

    def test_empty_string(self):
        self.assertEqual(self.parser.clean_company(""), "")


class TestCleanAddress(unittest.TestCase):
    """Tests for ReceiptParser.clean_address()."""

    def setUp(self):
        self.parser = ReceiptParser()

    def test_basic_cleaning(self):
        result = self.parser.clean_address("  123 Main St  ")
        self.assertIn("123", result)

    def test_empty_string(self):
        self.assertEqual(self.parser.clean_address(""), "")

    def test_none_input(self):
        self.assertEqual(self.parser.clean_address(None), "")


class TestStandardizeDate(unittest.TestCase):
    """Tests for ReceiptParser.standardize_date()."""

    def setUp(self):
        self.parser = ReceiptParser()

    def test_dd_mm_yyyy(self):
        result = self.parser.standardize_date("25/12/2023")
        self.assertEqual(result, "2023-12-25")

    def test_yyyy_mm_dd(self):
        result = self.parser.standardize_date("2023-06-15")
        self.assertEqual(result, "2023-06-15")

    def test_dd_month_yyyy(self):
        result = self.parser.standardize_date("15 Jan 2024")
        self.assertEqual(result, "2024-01-15")

    def test_mm_dd_yyyy(self):
        result = self.parser.standardize_date("12/25/2023")
        # Could parse as DD/MM or MM/DD — either is acceptable
        self.assertIn("2023", result)

    def test_short_year(self):
        result = self.parser.standardize_date("25/12/23")
        # Should attempt to parse the 2-digit year
        self.assertIsInstance(result, str)

    def test_empty_string(self):
        self.assertEqual(self.parser.standardize_date(""), "")

    def test_unparseable_date(self):
        result = self.parser.standardize_date("not a date")
        self.assertEqual(result, "not a date")  # Returns original

    def test_none_input(self):
        self.assertEqual(self.parser.standardize_date(None), "")


class TestCleanTotal(unittest.TestCase):
    """Tests for ReceiptParser.clean_total()."""

    def setUp(self):
        self.parser = ReceiptParser()

    def test_basic_amount(self):
        result = self.parser.clean_total("10.80")
        self.assertEqual(result, "10.80")

    def test_dollar_sign(self):
        result = self.parser.clean_total("$10.80")
        self.assertEqual(result, "10.80")

    def test_currency_prefix(self):
        result = self.parser.clean_total("RM 45.50")
        self.assertEqual(result, "45.50")

    def test_thousands_comma(self):
        result = self.parser.clean_total("1,234.56")
        self.assertEqual(result, "1234.56")

    def test_ocr_O_correction(self):
        result = self.parser.clean_total("1O.8O")
        self.assertEqual(result, "10.80")

    def test_ocr_l_correction(self):
        result = self.parser.clean_total("l5.00")
        self.assertEqual(result, "15.00")

    def test_comma_as_decimal(self):
        result = self.parser.clean_total("10,80")
        self.assertEqual(result, "10.80")

    def test_empty_string(self):
        result = self.parser.clean_total("")
        self.assertEqual(result, "0.00")

    def test_none_input(self):
        result = self.parser.clean_total(None)
        self.assertEqual(result, "0.00")

    def test_non_numeric(self):
        result = self.parser.clean_total("abc")
        self.assertEqual(result, "0.00")

    def test_formatting(self):
        result = self.parser.clean_total("10.8")
        self.assertEqual(result, "10.80")


class TestValidateTotal(unittest.TestCase):
    """Tests for ReceiptParser.validate_total()."""

    def setUp(self):
        self.parser = ReceiptParser()

    def test_valid_total(self):
        self.assertTrue(self.parser.validate_total("10.80"))

    def test_zero_total(self):
        self.assertFalse(self.parser.validate_total("0.00"))

    def test_negative_total(self):
        self.assertFalse(self.parser.validate_total("-5.00"))

    def test_very_large_total(self):
        self.assertFalse(self.parser.validate_total("99999999.00"))

    def test_non_numeric(self):
        self.assertFalse(self.parser.validate_total("abc"))


class TestValidateDate(unittest.TestCase):
    """Tests for ReceiptParser.validate_date()."""

    def setUp(self):
        self.parser = ReceiptParser()

    def test_valid_iso_date(self):
        self.assertTrue(self.parser.validate_date("2023-12-25"))

    def test_invalid_date_format(self):
        self.assertFalse(self.parser.validate_date("25/12/2023"))

    def test_empty_string(self):
        self.assertFalse(self.parser.validate_date(""))

    def test_none_input(self):
        self.assertFalse(self.parser.validate_date(None))


class TestParse(unittest.TestCase):
    """Tests for ReceiptParser.parse() end-to-end."""

    def setUp(self):
        self.parser = ReceiptParser()

    def test_full_parse(self):
        raw = {
            "company": "  book store inc  ",
            "date": "25/12/2023",
            "address": "123 Main St",
            "total": "$10.80",
        }
        result = self.parser.parse(raw)
        self.assertEqual(result["company"], "BOOK STORE INC")
        self.assertEqual(result["date"], "2023-12-25")
        self.assertIn("123", result["address"])
        self.assertEqual(result["total"], "10.80")

    def test_parse_empty_fields(self):
        raw = {"company": "", "date": "", "address": "", "total": ""}
        result = self.parser.parse(raw)
        self.assertEqual(result["company"], "")
        self.assertEqual(result["date"], "")
        self.assertEqual(result["address"], "")
        self.assertEqual(result["total"], "0.00")

    def test_parse_missing_keys(self):
        raw = {}
        result = self.parser.parse(raw)
        self.assertIn("company", result)
        self.assertIn("date", result)

    def test_parse_with_ocr_errors(self):
        raw = {
            "company": "GOLDEN PALACE",
            "date": "15/06/2023",
            "address": "45 JLN SULTAN",
            "total": "1O.8O",  # OCR confusion
        }
        result = self.parser.parse(raw)
        self.assertEqual(result["total"], "10.80")


class TestOCRCorrections(unittest.TestCase):
    """Tests for internal OCR correction logic."""

    def setUp(self):
        self.parser = ReceiptParser()

    def test_O_to_0_in_context(self):
        result = self.parser._correct_ocr_numbers("1O.5O")
        self.assertEqual(result, "10.50")

    def test_l_to_1_in_context(self):
        result = self.parser._correct_ocr_numbers("l5.00")
        self.assertEqual(result, "15.00")

    def test_no_correction_without_numeric_context(self):
        result = self.parser._correct_ocr_numbers("TOTAL")
        # Should NOT change letters that aren't in numeric context
        self.assertEqual(result, "TOTAL")

    def test_mixed_corrections(self):
        result = self.parser._correct_ocr_numbers("l0O.SO")
        # l→1, O→0, S→5 depending on context
        self.assertIn("1", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
