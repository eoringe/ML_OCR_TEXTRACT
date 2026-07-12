import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import torch
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizerFast

def test_inference():
    model_name = "npnam693/layoutlm-sroie"
    print("Loading model and tokenizer...")
    tokenizer = LayoutLMTokenizerFast.from_pretrained(model_name)
    model = LayoutLMForTokenClassification.from_pretrained(model_name)
    
    print("Model loaded successfully!")
    print("Labels (id2label):", model.config.id2label)
    
    # SROIE LayoutLM requires words as a single list, and boxes scaled to [0, 1000]
    words = ["East", "Repair", "Inc.", "1912", "Harvest", "Lane", "Total", "154.06"]
    boxes = [
        [100, 100, 200, 150],  # East
        [210, 100, 300, 150],  # Repair
        [310, 100, 400, 150],  # Inc.
        [100, 200, 150, 250],  # 1912
        [160, 200, 250, 250],  # Harvest
        [260, 200, 350, 250],  # Lane
        [100, 800, 180, 850],  # Total
        [200, 800, 300, 850],  # 154.06
    ]
    
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Align boxes with subtokens
    # word_ids returns a list mapping each token back to its word index in words
    word_ids = encoding.word_ids(batch_index=0)
    bbox_list = []
    for word_id in word_ids:
        if word_id is None:
            bbox_list.append([0, 0, 0, 0])
        else:
            bbox_list.append(boxes[word_id])
            
    # Add to encoding and convert to tensor
    encoding["bbox"] = torch.tensor([bbox_list])
    
    # We don't need offset_mapping for model input, so pop it
    encoding.pop("offset_mapping")
    
    with torch.no_grad():
        outputs = model(**encoding)
        
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    input_ids = encoding["input_ids"].squeeze().tolist()
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    print("\nToken Classification Predictions:")
    for token, pred_id, bbox in zip(tokens, predictions, bbox_list):
        label = model.config.id2label[pred_id]
        print(f"Token: '{token:12}' -> Label: '{label:10}' -> Box: {bbox}")

if __name__ == "__main__":
    test_inference()
