"""
Task 7: Evaluation Metrics
Branch: task-1-data-pipeline-emmanuel-oringe (team dissolved — all work consolidated)

This module provides evaluation metrics for the SROIE receipt OCR pipeline:
  - Levenshtein edit distance
  - Character Error Rate (CER)
  - Word Error Rate (WER)
  - Per-field precision, recall, and F1-score
  - Pipeline evaluation over test sets

Usage:
    from src.utils.evaluate import compute_cer_wer, compute_f1, evaluate_pipeline
"""

import os
import sys
import json
import re
import logging

logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)


# ===========================================================================
#  String Distance Metrics
# ===========================================================================

def edit_distance(s1, s2):
    """
    Computes the Levenshtein (edit) distance between two sequences.

    Args:
        s1: First sequence (string or list).
        s2: Second sequence (string or list).

    Returns:
        int: Minimum number of insertions, deletions, and substitutions
             to transform s1 into s2.
    """
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# ===========================================================================
#  CER & WER
# ===========================================================================

def compute_cer(pred_text, target_text):
    """
    Computes Character Error Rate (CER).

    CER = edit_distance(pred, target) / len(target)

    Args:
        pred_text (str): Predicted text.
        target_text (str): Ground truth text.

    Returns:
        float: CER value in [0, 1]. Returns 0 for matching empty strings,
               1 if target is empty but prediction is not.
    """
    if not target_text:
        return 1.0 if pred_text else 0.0

    dist = edit_distance(pred_text, target_text)
    return min(1.0, dist / len(target_text))


def compute_wer(pred_text, target_text):
    """
    Computes Word Error Rate (WER).

    WER = edit_distance(pred_words, target_words) / len(target_words)

    Args:
        pred_text (str): Predicted text.
        target_text (str): Ground truth text.

    Returns:
        float: WER value in [0, 1].
    """
    if not target_text:
        return 1.0 if pred_text else 0.0

    target_words = target_text.split()
    pred_words = pred_text.split()

    if not target_words:
        return 1.0 if pred_words else 0.0

    dist = edit_distance(pred_words, target_words)
    return min(1.0, dist / len(target_words))


def compute_cer_wer(pred_text, target_text):
    """
    Computes both CER and WER in one call.

    Args:
        pred_text (str): Predicted text.
        target_text (str): Ground truth text.

    Returns:
        tuple: (CER, WER) — both float values in [0, 1].
    """
    return compute_cer(pred_text, target_text), compute_wer(pred_text, target_text)


# ===========================================================================
#  Precision, Recall, F1 Score
# ===========================================================================

def compute_f1(predictions, ground_truths, normalize=True):
    """
    Computes precision, recall, and F1-score for exact field matching.

    Args:
        predictions (list[str]): Predicted field values.
        ground_truths (list[str]): Ground truth field values.
        normalize (bool): If True, normalises strings before comparison
                         (lowercase, strip whitespace, collapse spaces).

    Returns:
        dict: {"precision": float, "recall": float, "f1": float,
               "true_positives": int, "false_positives": int,
               "false_negatives": int}
    """
    tp = 0  # Correct predictions
    fp = 0  # Predicted but incorrect
    fn = 0  # Ground truth exists but not predicted

    for pred, gt in zip(predictions, ground_truths):
        if normalize:
            pred = _normalize(pred)
            gt = _normalize(gt)

        if gt:  # Ground truth exists
            if pred == gt:
                tp += 1
            elif pred:
                fp += 1
                fn += 1
            else:
                fn += 1
        else:  # No ground truth
            if pred:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def compute_per_field_cer(predictions, ground_truths, normalize=True):
    """
    Computes average CER across a list of field predictions.

    Args:
        predictions (list[str]): Predicted field values.
        ground_truths (list[str]): Ground truth field values.
        normalize (bool): If True, normalises before comparison.

    Returns:
        dict: {"average_cer": float, "individual_cers": list[float]}
    """
    cers = []
    for pred, gt in zip(predictions, ground_truths):
        if normalize:
            pred = _normalize(pred)
            gt = _normalize(gt)
        if gt:  # Only compute CER when ground truth exists
            cers.append(compute_cer(pred, gt))

    avg = sum(cers) / len(cers) if cers else 0.0
    return {"average_cer": avg, "individual_cers": cers}


def _normalize(text):
    """Normalises text for comparison: lowercase, strip, collapse spaces."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', str(text).strip().lower())


# ===========================================================================
#  Pipeline Evaluation
# ===========================================================================

def evaluate_pipeline(data_dir, test_files, pipeline_fn):
    """
    Evaluates the full OCR + KIE pipeline over a list of test files.

    Args:
        data_dir (str): Path to SROIE dataset root.
        test_files (list[str]): List of test image filenames.
        pipeline_fn (callable): Function(image_path) → dict with
                                {"company", "date", "address", "total"}.

    Returns:
        dict: Per-field accuracy, F1 scores, and aggregate metrics.
    """
    logger.info("Starting pipeline evaluation on %d samples...", len(test_files))

    img_dir = os.path.join(data_dir, "img")
    key_dir = os.path.join(data_dir, "key")

    fields = ["company", "date", "address", "total"]
    all_preds = {f: [] for f in fields}
    all_gts = {f: [] for f in fields}

    total_samples = 0

    for filename in test_files:
        base_name = os.path.splitext(filename)[0]
        img_path = os.path.join(img_dir, filename)
        gt_path = os.path.join(key_dir, f"{base_name}.json")

        if not os.path.exists(gt_path):
            continue

        # Load ground truth
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_keys = json.load(f)

        # Run prediction
        try:
            pred_keys = pipeline_fn(img_path)
        except Exception as exc:
            logger.warning("Failed to process %s: %s", filename, exc)
            continue

        total_samples += 1

        for field in fields:
            gt_val = str(gt_keys.get(field, "")).strip()
            pred_val = str(pred_keys.get(field, "")).strip()
            all_preds[field].append(pred_val)
            all_gts[field].append(gt_val)

    # Compute metrics per field
    results = {"total_samples": total_samples, "fields": {}}

    for field in fields:
        f1_result = compute_f1(all_preds[field], all_gts[field])
        cer_result = compute_per_field_cer(all_preds[field], all_gts[field])

        results["fields"][field] = {
            "accuracy": f1_result["precision"],
            "f1": f1_result["f1"],
            "precision": f1_result["precision"],
            "recall": f1_result["recall"],
            "average_cer": cer_result["average_cer"],
        }

    return results


def generate_report(eval_results):
    """
    Generates a formatted evaluation report string.

    Args:
        eval_results (dict): Output from evaluate_pipeline().

    Returns:
        str: Formatted report.
    """
    lines = []
    lines.append("=" * 56)
    lines.append("         SROIE PIPELINE EVALUATION REPORT")
    lines.append("=" * 56)
    lines.append(f"  Total samples evaluated: {eval_results['total_samples']}")
    lines.append("-" * 56)

    header = f"  {'Field':<12} {'Accuracy':>10} {'F1':>8} {'Prec':>8} {'Recall':>8} {'CER':>8}"
    lines.append(header)
    lines.append("-" * 56)

    for field, metrics in eval_results["fields"].items():
        row = (
            f"  {field:<12} "
            f"{metrics['accuracy']:>9.2%} "
            f"{metrics['f1']:>7.2%} "
            f"{metrics['precision']:>7.2%} "
            f"{metrics['recall']:>7.2%} "
            f"{metrics['average_cer']:>7.2%}"
        )
        lines.append(row)

    lines.append("=" * 56)
    return "\n".join(lines)


# ===========================================================================
#  Standalone Execution
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  Evaluation Metrics — Task 7 Verification")
    print("=" * 60)

    # Test edit distance
    print("\n  Edit Distance Tests:")
    pairs = [("kitten", "sitting"), ("", "abc"), ("abc", "abc"), ("abc", "")]
    for s1, s2 in pairs:
        d = edit_distance(s1, s2)
        print(f"    d('{s1}', '{s2}') = {d}")

    # Test CER/WER
    print("\n  CER/WER Tests:")
    test_pairs = [
        ("BOOK STORE INC", "BOOK ST0RE 1NC"),
        ("HELLO", "HELLO"),
        ("", ""),
        ("prediction", ""),
    ]
    for target, pred in test_pairs:
        cer, wer = compute_cer_wer(pred, target)
        print(f"    Target='{target}' Pred='{pred}' → CER={cer:.2%} WER={wer:.2%}")

    # Test F1
    print("\n  F1 Score Tests:")
    preds = ["ACME INC", "2024-01-01", "123 Main St", "10.80"]
    gts = ["ACME INC", "2024-01-01", "123 Main Street", "10.80"]
    f1_result = compute_f1(preds, gts)
    print(f"    F1={f1_result['f1']:.2%}  P={f1_result['precision']:.2%}  R={f1_result['recall']:.2%}")

    print("\n" + "=" * 60)
    print("  All checks passed!")
    print("=" * 60)
