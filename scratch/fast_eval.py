"""Fast eval on first 10 test images with per-sample mismatch reporting."""
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

_, _, test_files = split_dataset(data_dir)
eval_files = test_files[:10]
print(f"Running pipeline on: {eval_files}")

recognizer = TextRecognizer(use_easyocr=True)
detector = TextDetector()
extractor = KeyExtractor()
parser = ReceiptParser()

fields = ["company", "date", "address", "total"]
correct = {f: 0 for f in fields}
total_count = 0

for i, fname in enumerate(eval_files):
    img_path = os.path.join(img_dir, fname)
    key_path = os.path.join(key_dir, fname.replace(".jpg", ".json"))
    
    if not os.path.exists(img_path) or not os.path.exists(key_path):
        continue
    
    with open(key_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    
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
    print(f"\n--- File: {fname} ---")
    for f in fields:
        pred = parsed.get(f, "").strip()
        truth = gt.get(f, "").strip()
        is_correct = pred.lower() == truth.lower()
        if is_correct:
            correct[f] += 1
        print(f"  {f.upper():8s} | Match: {str(is_correct):5s} | Pred: '{pred}' | GT: '{truth}'")

print(f"\n{'='*70}")
print(f"ACCURACY REPORT ({total_count} images)")
print(f"{'='*70}")
for f in fields:
    acc = (correct[f] / total_count * 100) if total_count > 0 else 0
    print(f"  {f:12s}: {acc:6.1f}%  ({correct[f]}/{total_count})")
