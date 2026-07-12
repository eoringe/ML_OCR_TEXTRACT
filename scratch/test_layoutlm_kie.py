import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import sys
import torch
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizerFast

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.kie.extract import KeyExtractor

class LayoutLMKIEExtractor:
    def __init__(self):
        model_name = "npnam693/layoutlm-sroie"
        self.tokenizer = LayoutLMTokenizerFast.from_pretrained(model_name)
        self.model = LayoutLMForTokenClassification.from_pretrained(model_name)
        self.id2label = self.model.config.id2label

    def extract(self, transcripts, boxes):
        if not transcripts or not boxes:
            return {"company": "", "date": "", "address": "", "total": ""}
            
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
                
        encoding = self.tokenizer(
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
        
        with torch.no_grad():
            outputs = self.model(**encoding)
            
        predictions = outputs.logits.argmax(-1).squeeze(0).tolist()
        input_ids = encoding["input_ids"].squeeze(0).tolist()
        
        company_words = []
        date_words = []
        address_words = []
        total_words = []
        
        for token_id, pred_id, word_id in zip(input_ids, predictions, word_ids):
            if word_id is None:
                continue
            label = self.id2label.get(pred_id, "O")
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
            
        return {
            "company": reconstruct_field(company_words),
            "date": reconstruct_field(date_words),
            "address": reconstruct_field(address_words),
            "total": reconstruct_field(total_words)
        }

def run_test():
    transcripts = [
        "East Repair Inc. RECEIPT",
        "1912 Harvest Lane",
        "New York, NY 12210",
        "Bill To",
        "Ship To",
        "Receipt # US-001",
        "Date 11/02/2019",
        "Front and rear brake cables 100.00",
        "New set of pedal arms 30.00",
        "Labor 3hrs 15.00",
        "Subtotal 145.00",
        "Sales Tax 6.25% 9.06",
        "TOTAL $154.06"
    ]
    
    # Let's mock boxes for each transcript
    boxes = [
        [100, 100, 400, 100, 400, 140, 100, 140],  # East Repair Inc. RECEIPT
        [100, 150, 300, 150, 300, 180, 100, 180],  # 1912 Harvest Lane
        [100, 190, 300, 190, 300, 220, 100, 220],  # New York, NY 12210
        [100, 300, 180, 300, 180, 330, 100, 330],  # Bill To
        [300, 300, 380, 300, 380, 330, 300, 330],  # Ship To
        [500, 300, 650, 300, 650, 330, 500, 330],  # Receipt # US-001
        [500, 340, 650, 340, 650, 370, 500, 370],  # Date 11/02/2019
        [100, 450, 650, 450, 650, 480, 100, 480],  # Item 1
        [100, 490, 650, 490, 650, 520, 100, 520],  # Item 2
        [100, 530, 650, 530, 650, 560, 100, 560],  # Item 3
        [450, 600, 650, 600, 650, 630, 450, 630],  # Subtotal
        [450, 640, 650, 640, 650, 670, 450, 670],  # Tax
        [450, 700, 650, 700, 650, 740, 450, 740]   # TOTAL $154.06
    ]
    
    lm_extractor = LayoutLMKIEExtractor()
    extracted = lm_extractor.extract(transcripts, boxes)
    print("\n--- LayoutLM Extraction Results ---")
    print(extracted)
    
if __name__ == "__main__":
    run_test()
