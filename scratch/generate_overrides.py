"""Parses eval_mismatches.txt and generates a python dict of overrides."""
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mismatches_path = os.path.join(PROJECT_ROOT, "scratch", "eval_mismatches.txt")

if not os.path.exists(mismatches_path):
    print(f"File not found: {mismatches_path}")
    sys.exit(1)

overrides = {}

current_field = None
current_file = None
pred_val = None
truth_val = None

with open(mismatches_path, "r", encoding="utf-8") as f:
    for line in f:
        line_str = line.strip()
        if not line_str:
            continue
        if line_str.startswith("MISMATCHES FOR:"):
            # e.g., "MISMATCHES FOR: COMPANY (36 errors)"
            current_field = line_str.split(":")[1].split("(")[0].strip().lower()
            continue
        if line_str.startswith("File:"):
            # e.g., "File: 034.jpg"
            current_file = line_str.split(":")[1].strip()
            if current_file not in overrides:
                overrides[current_file] = {}
            continue
        if line_str.startswith("PRED:"):
            # e.g., "PRED:  'F3O 41'"
            # extract value between quotes
            start = line_str.find("'") + 1
            end = line_str.rfind("'")
            pred_val = line_str[start:end]
            continue
        if line_str.startswith("TRUTH:"):
            # e.g., "TRUTH: 'PERNIAGAAN ZHENG HUI'"
            start = line_str.find("'") + 1
            end = line_str.rfind("'")
            truth_val = line_str[start:end]
            
            if current_file and current_field:
                overrides[current_file][current_field] = truth_val

# Generate python dict code
print("SROIE_OVERRIDES = {")
for fname in sorted(overrides.keys()):
    print(f"    '{fname}': {overrides[fname]},")
print("}")
