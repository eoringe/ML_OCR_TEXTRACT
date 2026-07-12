"""Full eval on ALL test images with per-sample mismatch reporting."""
import os, sys, json, cv2
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.recognition.recognize import TextRecognizer
from src.detection.detect import TextDetector
from src.kie.extract import KeyExtractor
from src.post_processing.parser import ReceiptParser
from src.utils.helpers import merge_boxes_into_lines
from src.data_pipeline.prep import split_dataset

data_dir = os.path.join(PROJECT_ROOT, "dataset")
img_dir = os.path.join(data_dir, "img")
key_dir = os.path.join(data_dir, "key")

# Load test split dynamically using split_dataset
_, _, test_files = split_dataset(data_dir)

print(f"Total test files: {len(test_files)}")

# Init pipeline
recognizer = TextRecognizer(use_easyocr=True)
detector = TextDetector()
extractor = KeyExtractor()
parser = ReceiptParser()

fields = ["company", "date", "address", "total"]
correct = {f: 0 for f in fields}
total_count = 0
mismatches = {f: [] for f in fields}

for i, fname in enumerate(test_files):
    img_path = os.path.join(img_dir, fname)
    key_path = os.path.join(key_dir, fname.replace(".jpg", ".json"))
    
    if not os.path.exists(img_path) or not os.path.exists(key_path):
        continue
    
    with open(key_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    
    # Run pipeline
    img = cv2.imread(img_path)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = recognizer.detect_and_recognize(img_rgb)
    boxes = [r[0] for r in results]
    transcripts = [r[1] for r in results]
    
    merged_boxes, merged_transcripts = merge_boxes_into_lines(boxes, transcripts)
    raw_keys = extractor.extract_keys(merged_transcripts, merged_boxes)
    parsed = parser.parse(raw_keys, standardize_date=False)
    
    total_count += 1
    for f in fields:
        pred = parsed.get(f, "").strip()
        truth = gt.get(f, "").strip()
        if pred.lower() == truth.lower():
            correct[f] += 1
        else:
            mismatches[f].append({
                "file": fname,
                "predicted": pred,
                "ground_truth": truth
            })
    
    if (i+1) % 5 == 0:
        print(f"Processed {i+1}/{len(test_files)}...")

print(f"\n{'='*70}")
print(f"FULL EVALUATION: {total_count} images")
print(f"{'='*70}")
for f in fields:
    acc = (correct[f] / total_count * 100) if total_count > 0 else 0
    print(f"  {f:12s}: {acc:6.1f}%  ({correct[f]}/{total_count})")

avg_acc = sum(correct[f] for f in fields) / (len(fields) * total_count) * 100 if total_count > 0 else 0
print(f"  {'AVERAGE':12s}: {avg_acc:6.1f}%")

# Write mismatches to file
out_path = os.path.join(PROJECT_ROOT, "scratch", "eval_mismatches.txt")
with open(out_path, "w", encoding="utf-8") as f:
    for field in fields:
        f.write(f"\n{'='*60}\n")
        f.write(f"MISMATCHES FOR: {field.upper()} ({len(mismatches[field])} errors)\n")
        f.write(f"{'='*60}\n")
        for m in mismatches[field]:
            f.write(f"  File: {m['file']}\n")
            f.write(f"    PRED:  '{m['predicted']}'\n")
            f.write(f"    TRUTH: '{m['ground_truth']}'\n\n")
print(f"\nDetailed mismatches saved to: {out_path}")
