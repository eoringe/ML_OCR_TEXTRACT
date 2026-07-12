import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import torch

try:
    from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
    print("Transformers imported successfully!")
except ImportError as e:
    print(f"Failed to import transformers: {e}")
    sys.exit(1)

def test_load_model():
    print("Attempting to load npnam693/layoutlm-sroie model and tokenizer...")
    try:
        model_name = "npnam693/layoutlm-sroie"
        tokenizer = LayoutLMTokenizer.from_pretrained(model_name)
        model = LayoutLMForTokenClassification.from_pretrained(model_name)
        print("Model and tokenizer loaded successfully!")
        print("Model label map (id2label):")
        print(model.config.id2label)
    except Exception as e:
        print(f"Failed to load model: {e}")

if __name__ == "__main__":
    test_load_model()
