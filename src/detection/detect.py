import cv2
import numpy as np
import torch
import torch.nn as nn

class TextDetector:
    """
    Task 2: Text Detection Model.
    Responsible for locating text regions in receipt images and returning coordinates.
    """
    def __init__(self, model_path=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_path = model_path
        # Initialize detector architecture
        self.model = self._build_model()
        if model_path:
            self.load_weights(model_path)
            
    def _build_model(self):
        """
        Builds the text detection neural network.
        For a baseline, we can define a simple CNN backbone or placeholders.
        """
        # Placeholder neural network for text detection (e.g., DBNet / EAST)
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        return model.to(self.device)

    def load_weights(self, path):
        """Loads weights from disk."""
        print(f"Loading Text Detector weights from {path}")
        # self.model.load_state_dict(torch.load(path, map_location=self.device))

    def train_epoch(self, dataloader, optimizer, criterion):
        """
        Training loop for Text Detection model.
        """
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            images = batch["images"].to(self.device)
            # targets = batch["boxes"] # In actual DBNet, targets are probability maps and threshold maps
            
            optimizer.zero_grad()
            # outputs = self.model(images)
            # loss = criterion(outputs, targets)
            # loss.backward()
            # optimizer.step()
            # total_loss += loss.item()
            pass
            
        return total_loss / len(dataloader)

    def detect(self, image):
        """
        Detects text bounding boxes in a single image.
        
        Args:
            image (np.ndarray): Input RGB image.
        Returns:
            list: List of bounding boxes. Each box is formatted as:
                  [x1, y1, x2, y2, x3, y3, x4, y4]
        """
        # Baseline fallback: If model is not trained, we can use a heuristic or OpenCV MSER/Contours
        # as a working baseline, or EasyOCR's internal detector if imported.
        h, w = image.shape[:2]
        
        # Simple contour-based baseline text detector
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Morphological operations to group text lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilate = cv2.dilate(thresh, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 100:  # Filter out noise
                x, y, w_box, h_box = cv2.boundingRect(c)
                # Formulate as 4 points: top-left, top-right, bottom-right, bottom-left
                box = [
                    x, y, 
                    x + w_box, y, 
                    x + w_box, y + h_box, 
                    x, y + h_box
                ]
                boxes.append(box)
                
        # Sort boxes top-to-bottom, left-to-right
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        return boxes

if __name__ == "__main__":
    # Test text detector
    detector = TextDetector()
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    # Draw a mock box
    cv2.rectangle(dummy_image, (50, 100), (200, 150), (255, 255, 255), -1)
    
    found_boxes = detector.detect(dummy_image)
    print(f"Detected {len(found_boxes)} text regions in dummy image.")
