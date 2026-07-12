"""
Unit tests for Task 7: Evaluation Metrics
Run with: python -m pytest tests/test_evaluate.py -v
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.evaluate import (
    edit_distance,
    compute_cer,
    compute_wer,
    compute_cer_wer,
    compute_f1,
    compute_per_field_cer,
    generate_report,
)


# ======================================================================
#  Test: Edit Distance
# ======================================================================

class TestEditDistance(unittest.TestCase):
    """Tests for edit_distance()."""

    def test_identical_strings(self):
        self.assertEqual(edit_distance("hello", "hello"), 0)

    def test_empty_strings(self):
        self.assertEqual(edit_distance("", ""), 0)

    def test_one_empty(self):
        self.assertEqual(edit_distance("abc", ""), 3)
        self.assertEqual(edit_distance("", "abc"), 3)

    def test_single_substitution(self):
        self.assertEqual(edit_distance("cat", "bat"), 1)

    def test_single_insertion(self):
        self.assertEqual(edit_distance("cat", "cats"), 1)

    def test_single_deletion(self):
        self.assertEqual(edit_distance("cats", "cat"), 1)

    def test_known_distance(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)

    def test_symmetric(self):
        d1 = edit_distance("abc", "def")
        d2 = edit_distance("def", "abc")
        self.assertEqual(d1, d2)

    def test_word_sequences(self):
        # Edit distance on word lists
        words1 = ["hello", "world"]
        words2 = ["hello", "there", "world"]
        self.assertEqual(edit_distance(words1, words2), 1)


# ======================================================================
#  Test: CER
# ======================================================================

class TestComputeCER(unittest.TestCase):
    """Tests for compute_cer()."""

    def test_identical_strings(self):
        cer = compute_cer("hello", "hello")
        self.assertAlmostEqual(cer, 0.0)

    def test_completely_different(self):
        cer = compute_cer("abc", "xyz")
        self.assertGreater(cer, 0.0)

    def test_empty_target_with_prediction(self):
        cer = compute_cer("prediction", "")
        self.assertAlmostEqual(cer, 1.0)

    def test_empty_target_empty_prediction(self):
        cer = compute_cer("", "")
        self.assertAlmostEqual(cer, 0.0)

    def test_partial_match(self):
        cer = compute_cer("hell", "hello")
        self.assertGreater(cer, 0.0)
        self.assertLess(cer, 1.0)

    def test_cer_bounded(self):
        cer = compute_cer("abcdefghij", "xyz")
        self.assertLessEqual(cer, 1.0)


# ======================================================================
#  Test: WER
# ======================================================================

class TestComputeWER(unittest.TestCase):
    """Tests for compute_wer()."""

    def test_identical_sentences(self):
        wer = compute_wer("hello world", "hello world")
        self.assertAlmostEqual(wer, 0.0)

    def test_one_word_different(self):
        wer = compute_wer("hello world", "hello there")
        self.assertAlmostEqual(wer, 0.5)

    def test_completely_different(self):
        wer = compute_wer("a b c", "x y z")
        self.assertAlmostEqual(wer, 1.0)

    def test_empty_target_with_prediction(self):
        wer = compute_wer("some words", "")
        self.assertAlmostEqual(wer, 1.0)

    def test_empty_both(self):
        wer = compute_wer("", "")
        self.assertAlmostEqual(wer, 0.0)

    def test_extra_word(self):
        wer = compute_wer("hello beautiful world", "hello world")
        self.assertGreater(wer, 0.0)


# ======================================================================
#  Test: CER + WER Combined
# ======================================================================

class TestComputeCERWER(unittest.TestCase):
    """Tests for compute_cer_wer()."""

    def test_returns_tuple(self):
        result = compute_cer_wer("hello", "hello")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_identical_returns_zeros(self):
        cer, wer = compute_cer_wer("hello world", "hello world")
        self.assertAlmostEqual(cer, 0.0)
        self.assertAlmostEqual(wer, 0.0)

    def test_cer_wer_consistency(self):
        cer, wer = compute_cer_wer("hello", "hello world")
        # Both should be in [0, 1]
        self.assertGreaterEqual(cer, 0.0)
        self.assertLessEqual(cer, 1.0)
        self.assertGreaterEqual(wer, 0.0)
        self.assertLessEqual(wer, 1.0)


# ======================================================================
#  Test: F1 Score
# ======================================================================

class TestComputeF1(unittest.TestCase):
    """Tests for compute_f1()."""

    def test_perfect_predictions(self):
        preds = ["ACME", "2024-01-01", "123 Main St", "10.80"]
        gts = ["ACME", "2024-01-01", "123 Main St", "10.80"]
        result = compute_f1(preds, gts)
        self.assertAlmostEqual(result["f1"], 1.0)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)

    def test_all_wrong(self):
        preds = ["wrong1", "wrong2"]
        gts = ["correct1", "correct2"]
        result = compute_f1(preds, gts)
        self.assertAlmostEqual(result["f1"], 0.0)

    def test_partial_match(self):
        preds = ["ACME", "wrong"]
        gts = ["ACME", "correct"]
        result = compute_f1(preds, gts)
        self.assertGreater(result["f1"], 0.0)
        self.assertLess(result["f1"], 1.0)

    def test_empty_predictions(self):
        preds = ["", ""]
        gts = ["ACME", "2024"]
        result = compute_f1(preds, gts)
        self.assertAlmostEqual(result["precision"], 0.0)

    def test_normalized_comparison(self):
        preds = ["  ACME  ", "acme"]
        gts = ["acme", "  ACME  "]
        result = compute_f1(preds, gts, normalize=True)
        self.assertAlmostEqual(result["f1"], 1.0)

    def test_without_normalization(self):
        preds = ["ACME"]
        gts = ["acme"]
        result = compute_f1(preds, gts, normalize=False)
        self.assertAlmostEqual(result["f1"], 0.0)

    def test_result_structure(self):
        result = compute_f1(["a"], ["a"])
        self.assertIn("precision", result)
        self.assertIn("recall", result)
        self.assertIn("f1", result)
        self.assertIn("true_positives", result)
        self.assertIn("false_positives", result)
        self.assertIn("false_negatives", result)


# ======================================================================
#  Test: Per-Field CER
# ======================================================================

class TestComputePerFieldCER(unittest.TestCase):
    """Tests for compute_per_field_cer()."""

    def test_perfect_match(self):
        result = compute_per_field_cer(["hello"], ["hello"])
        self.assertAlmostEqual(result["average_cer"], 0.0)

    def test_partial_match(self):
        result = compute_per_field_cer(["hell"], ["hello"])
        self.assertGreater(result["average_cer"], 0.0)

    def test_empty_ground_truth_skipped(self):
        result = compute_per_field_cer(["pred", ""], ["gt", ""])
        # Empty GT should be skipped
        self.assertEqual(len(result["individual_cers"]), 1)

    def test_multiple_fields(self):
        preds = ["ACME", "2024-01-01", "10.80"]
        gts = ["ACME", "2024-01-01", "10.80"]
        result = compute_per_field_cer(preds, gts)
        self.assertAlmostEqual(result["average_cer"], 0.0)
        self.assertEqual(len(result["individual_cers"]), 3)

    def test_result_structure(self):
        result = compute_per_field_cer(["a"], ["b"])
        self.assertIn("average_cer", result)
        self.assertIn("individual_cers", result)


# ======================================================================
#  Test: Report Generation
# ======================================================================

class TestGenerateReport(unittest.TestCase):
    """Tests for generate_report()."""

    def test_generates_string(self):
        eval_results = {
            "total_samples": 10,
            "fields": {
                "company": {"accuracy": 0.8, "f1": 0.75, "precision": 0.8, "recall": 0.7, "average_cer": 0.1},
                "date": {"accuracy": 0.9, "f1": 0.85, "precision": 0.9, "recall": 0.8, "average_cer": 0.05},
                "address": {"accuracy": 0.6, "f1": 0.55, "precision": 0.6, "recall": 0.5, "average_cer": 0.2},
                "total": {"accuracy": 0.95, "f1": 0.9, "precision": 0.95, "recall": 0.85, "average_cer": 0.03},
            },
        }
        report = generate_report(eval_results)
        self.assertIsInstance(report, str)
        self.assertIn("company", report)
        self.assertIn("total", report)
        self.assertIn("10", report)  # total samples


if __name__ == "__main__":
    unittest.main(verbosity=2)
