import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class ReceiptDataset(Dataset):
    """
    Custom PyTorch Dataset for SROIE Receipt Images and OCR Annotations.
    """
    def __init__(self, data_dir, image_files, transform=None):
        """
        Args:
            data_dir (str): Path to the SROIE dataset folder.
            image_files (list): List of image filenames (e.g. ['000.jpg', '001.jpg']).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.image_files = image_files
        self.transform = transform
        
        self.img_dir = os.path.join(data_dir, "img")
        self.box_dir = os.path.join(data_dir, "box")
        self.key_dir = os.path.join(data_dir, "key")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        base_name = os.path.splitext(img_name)[0]
        
        # Load image
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load OCR / Bounding box annotations (Task 1 & 2 representation)
        box_path = os.path.join(self.box_dir, f"{base_name}.txt")
        boxes, transcripts = self._load_boxes(box_path)
        
        # Load Key Information labels (Task 4 representation)
        key_path = os.path.join(self.key_dir, f"{base_name}.json")
        keys = self._load_keys(key_path)
        
        # Preprocessing & Augmentation (Task 1 main job)
        processed_image = preprocess_image(image)
        
        if self.transform:
            processed_image = self.transform(processed_image)
            
        sample = {
            "image": processed_image,
            "original_image": image,
            "boxes": boxes,
            "transcripts": transcripts,
            "keys": keys,
            "filename": img_name
        }
        
        return sample

    def _load_boxes(self, filepath):
        """
        Loads bounding box files containing: x1,y1,x2,y2,x3,y3,x4,y4,transcription
        """
        boxes = []
        transcripts = []
        if not os.path.exists(filepath):
            return np.array(boxes), transcripts
            
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",", 8)
                if len(parts) >= 9:
                    coords = [int(float(x)) for x in parts[:8]]
                    transcript = parts[8]
                    boxes.append(coords)
                    transcripts.append(transcript)
        return np.array(boxes), transcripts

    def _load_keys(self, filepath):
        """
        Loads semantic ground truth keys: company, date, address, total
        """
        import json
        if not os.path.exists(filepath):
            return {"company": "", "date": "", "address": "", "total": ""}
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            try:
                return json.load(f)
            except Exception:
                return {"company": "", "date": "", "address": "", "total": ""}

def preprocess_image(image, target_size=(640, 640)):
    """
    Task 1: Preprocesses the receipt image.
    Applies resizing, normalization, and optional deskewing/binarization.
    
    Args:
        image (np.ndarray): Input RGB image.
        target_size (tuple): Target resize dimensions.
    Returns:
        np.ndarray: Preprocessed image.
    """
    # 1. Resize maintaining aspect ratio or padding
    h, w = image.shape[:2]
    # Simple resize for baseline; advanced preprocessors should pad to maintain ratio
    resized = cv2.resize(image, target_size)
    
    # 2. Binarization / Contrast Enhancement (Optional)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    # Apply Otsu's thresholding or adaptive thresholding
    binarized = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Return 3-channel stack of binarized image for compatibility with CNN backbones
    processed = cv2.merge([binarized, binarized, binarized])
    return processed

def split_dataset(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits the dataset files evenly.
    """
    img_dir = os.path.join(data_dir, "img")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory {img_dir} does not exist. Run scripts/download_dataset.py first.")
        
    all_images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    # Set seed for reproducibility
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(all_images))
    
    train_end = int(len(all_images) * train_ratio)
    val_end = train_end + int(len(all_images) * val_ratio)
    
    train_files = [all_images[i] for i in shuffled_indices[:train_end]]
    val_files = [all_images[i] for i in shuffled_indices[train_end:val_end]]
    test_files = [all_images[i] for i in shuffled_indices[val_end:]]
    
    return train_files, val_files, test_files

def get_dataloaders(data_dir, batch_size=8, target_size=(640, 640)):
    """
    Helper function to get PyTorch DataLoaders for Train, Val, and Test sets.
    """
    train_files, val_files, test_files = split_dataset(data_dir)
    
    train_ds = ReceiptDataset(data_dir, train_files)
    val_ds = ReceiptDataset(data_dir, val_files)
    test_ds = ReceiptDataset(data_dir, test_files)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader

def collate_fn(batch):
    """
    Custom collate function to handle variable sized lists of boxes and annotations.
    """
    images = torch.stack([torch.tensor(item["image"], dtype=torch.float32).permute(2, 0, 1) / 255.0 for item in batch])
    original_images = [item["original_image"] for item in batch]
    boxes = [item["boxes"] for item in batch]
    transcripts = [item["transcripts"] for item in batch]
    keys = [item["keys"] for item in batch]
    filenames = [item["filename"] for item in batch]
    
    return {
        "images": images,
        "original_images": original_images,
        "boxes": boxes,
        "transcripts": transcripts,
        "keys": keys,
        "filenames": filenames
    }

if __name__ == "__main__":
    # Test script loading
    try:
        train_files, val_files, test_files = split_dataset("dataset")
        print(f"Dataset Split Success:")
        print(f"  Train: {len(train_files)} files")
        print(f"  Val:   {len(val_files)} files")
        print(f"  Test:  {len(test_files)} files")
    except Exception as e:
        print(f"Test loading failed: {e}. (Ensure SROIE dataset is downloaded first)")
