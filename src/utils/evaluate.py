import os
import sys
import json
import re

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Helper function to compute edit distance (Levenshtein)
def edit_distance(s1, s2):
    """
    Computes Levenshtein distance between two sequences.
    """
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def compute_cer_wer(pred_text, target_text):
    """
    Computes Character Error Rate (CER) and Word Error Rate (WER).
    """
    if not target_text:
        return (1.0, 1.0) if pred_text else (0.0, 0.0)
        
    cer_dist = edit_distance(pred_text, target_text)
    cer = cer_dist / len(target_text)
    
    target_words = target_text.split()
    pred_words = pred_text.split()
    
    wer_dist = edit_distance(pred_words, target_words)
    wer = wer_dist / max(1, len(target_words))
    
    return min(1.0, cer), min(1.0, wer)

def evaluate_pipeline(data_dir, test_files, pipeline_fn):
    """
    Evaluates the full OCR + KIE pipeline over a list of test files.
    
    Args:
        data_dir (str): Path to SROIE dataset.
        test_files (list): List of test image filenames.
        pipeline_fn (callable): A function that takes an image path and returns parsed keys.
    """
    print(f"Starting pipeline evaluation on {len(test_files)} samples...")
    
    total_samples = 0
    correct_fields = {"company": 0, "date": 0, "address": 0, "total": 0}
    field_counts = {"company": 0, "date": 0, "address": 0, "total": 0}
    
    img_dir = os.path.join(data_dir, "img")
    key_dir = os.path.join(data_dir, "key")
    
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
        except Exception as e:
            print(f"Failed to process sample {filename}: {e}")
            continue
            
        total_samples += 1
        
        # Evaluate each field (exact match or normalized match)
        for field in ["company", "date", "address", "total"]:
            gt_val = str(gt_keys.get(field, "")).strip().lower()
            pred_val = str(pred_keys.get(field, "")).strip().lower()
            
            # Simple cleaning for robust evaluation
            gt_clean = re.sub(r'\s+', ' ', gt_val)
            pred_clean = re.sub(r'\s+', ' ', pred_val)
            
            if gt_clean:
                field_counts[field] += 1
                if pred_clean == gt_clean:
                    correct_fields[field] += 1
                    
    # Print metrics
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Processed samples: {total_samples}")
    print("----------------------------------------------------")
    for field in ["company", "date", "address", "total"]:
        count = field_counts[field]
        correct = correct_fields[field]
        accuracy = (correct / count) * 100 if count > 0 else 0.0
        print(f"  Field '{field:<8}': Accuracy = {accuracy:>6.2f}% ({correct}/{count})")
    print("====================================================")
    
    return {f: (correct_fields[f] / field_counts[f] if field_counts[f] > 0 else 0) for f in field_counts}

if __name__ == "__main__":
    # Test metric calculations
    s1 = "BOOK STORE INC"
    s2 = "BOOK ST0RE 1NC"
    cer, wer = compute_cer_wer(s2, s1)
    print(f"Metric Test:")
    print(f"  Target: '{s1}'")
    print(f"  Pred:   '{s2}'")
    print(f"  CER:    {cer:.2%}")
