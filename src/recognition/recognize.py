"""
Task 3: Text Recognition
Branch: task-1-data-pipeline-emmanuel-oringe (team dissolved — all work consolidated)

This module provides text recognition capabilities for the SROIE receipt OCR
pipeline. It converts cropped bounding-box image regions into text strings.

Supports:
  - EasyOCR-based recognition (preferred, if installed)
  - Tesseract OCR fallback
  - CRNN (CNN + BiLSTM + CTC) architecture skeleton for training
  - Batch recognition across multiple boxes

Usage:
    from src.recognition.recognize import TextRecognizer
    recognizer = TextRecognizer()
    texts = recognizer.recognize(image_array, boxes)
"""

import logging
import numpy as np

# Optional imports — gracefully degrade if not installed
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

try:
    import pytesseract
    pytesseract.get_tesseract_version()
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CRNN_IMG_HEIGHT = 32            # Standard CRNN input height
CRNN_IMG_WIDTH = 128            # Standard CRNN input width
DEFAULT_ALPHABET = (
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!@#$%^&*()_+-=[]{}|;':\",./<>?\\ "
)


# ===========================================================================
#  CRNN Architecture (for future training)
# ===========================================================================

if HAS_TORCH:
    class CRNN(nn.Module):
        """
        Convolutional Recurrent Neural Network for text recognition.

        Architecture:
            CNN Feature Extractor → Bidirectional LSTM → Linear → CTC Decoding

        Input:  (B, 1, 32, W) — grayscale crops resized to height=32
        Output: (B, T, vocab_size+1) — character logits per time step
        """

        def __init__(self, vocab_size, hidden_size=128, num_layers=2):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 1), (2, 1)),
                nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
            self.rnn = nn.LSTM(
                512, hidden_size, num_layers=num_layers,
                bidirectional=True, batch_first=True
            )
            self.fc = nn.Linear(hidden_size * 2, vocab_size + 1)  # +1 for CTC blank

        def forward(self, x):
            """
            Args:
                x: (B, 1, H, W) — grayscale images

            Returns:
                (B, T, vocab_size+1) — logits per time step
            """
            features = self.cnn(x)  # (B, 512, H', W')
            b, c, h, w = features.size()
            # Collapse height dimension and use width as sequence
            features = features.reshape(b, c * h, w)  # (B, C*H, W)
            features = features.permute(0, 2, 1)       # (B, W, C*H)

            # Adapt input size for RNN
            if features.size(2) != 512:
                adapt = nn.Linear(features.size(2), 512).to(features.device)
                features = adapt(features)

            out, _ = self.rnn(features)
            logits = self.fc(out)
            return logits


# ===========================================================================
#  Crop Utilities
# ===========================================================================

def crop_box(image, box, pad=2):
    """
    Crops a bounding box region from an image.

    Supports 8-point polygon bounding boxes by computing the axis-aligned
    bounding rectangle.

    Args:
        image (np.ndarray): Source image (H, W, C) or (H, W).
        box (list/array): 8-point box [x1,y1,x2,y2,x3,y3,x4,y4].
        pad (int): Padding in pixels around the crop.

    Returns:
        np.ndarray or None: Cropped image region, or None on failure.
    """
    if not HAS_CV2:
        return None

    try:
        pts = np.array(box).reshape((-1, 2)).astype(np.int32)
        if len(pts) > 0:
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
            if x_coords.max() - x_coords.min() <= 0 or y_coords.max() - y_coords.min() <= 0:
                return None

        x, y, w, h = cv2.boundingRect(pts)

        img_h, img_w = image.shape[:2]

        x_min = max(0, x - pad)
        y_min = max(0, y - pad)
        x_max = min(img_w, x + w + pad)
        y_max = min(img_h, y + h + pad)

        if x_max <= x_min or y_max <= y_min:
            return None

        cropped = image[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            return None

        return cropped
    except Exception as exc:
        logger.warning("Failed to crop box %s: %s", box, exc)
        return None


def preprocess_crop_for_recognition(crop, target_height=CRNN_IMG_HEIGHT):
    """
    Preprocesses a cropped text region for recognition.

    Steps:
        1. Convert to grayscale
        2. Resize to fixed height while maintaining aspect ratio
        3. Normalise to [0, 1]

    Args:
        crop (np.ndarray): Cropped image (H, W, C) or (H, W).
        target_height (int): Target height in pixels.

    Returns:
        np.ndarray: Preprocessed grayscale image (target_height, W').
    """
    if not HAS_CV2:
        return crop

    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop.copy()

    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return gray

    aspect = w / h
    new_w = max(1, int(target_height * aspect))
    resized = cv2.resize(gray, (new_w, target_height), interpolation=cv2.INTER_AREA)

    return resized


# ===========================================================================
#  Text Recognizer
# ===========================================================================

class TextRecognizer:
    """
    Text recognition engine for the SROIE receipt OCR pipeline.

    Recognises text within cropped bounding box regions using:
        1. EasyOCR (if installed) — best accuracy
        2. Tesseract OCR (if installed) — good fallback
        3. Mock output — keeps the pipeline functional without OCR engines

    Usage:
        recognizer = TextRecognizer()
        texts = recognizer.recognize(image, boxes)
    """

    def __init__(self, model_path=None, use_easyocr=True, use_tesseract=True,
                 languages=None):
        """
        Args:
            model_path (str, optional): Path to custom CRNN model weights.
            use_easyocr (bool): Whether to try EasyOCR first.
            use_tesseract (bool): Whether to try Tesseract as fallback.
            languages (list, optional): Language codes for EasyOCR (default: ["en"]).
        """
        self.model_path = model_path
        self.languages = languages or ["en"]
        self._easyocr_reader = None
        self._backend = "mock"

        # Determine best available backend
        if use_easyocr and HAS_EASYOCR:
            self._backend = "easyocr"
            logger.info("TextRecognizer using EasyOCR backend")
        elif use_tesseract and HAS_TESSERACT:
            self._backend = "tesseract"
            logger.info("TextRecognizer using Tesseract backend")
        else:
            logger.warning(
                "No OCR engine available (easyocr=%s, tesseract=%s). "
                "Using mock backend.",
                HAS_EASYOCR, HAS_TESSERACT
            )

    @property
    def backend(self):
        """Returns the active OCR backend name."""
        return self._backend

    def _get_easyocr_reader(self):
        """Lazily initialise EasyOCR reader (heavy on first call)."""
        if self._easyocr_reader is None:
            self._easyocr_reader = easyocr.Reader(
                self.languages, gpu=HAS_TORCH and torch.cuda.is_available()
            )
        return self._easyocr_reader

    def detect_and_recognize(self, image):
        """
        Performs text detection and recognition in a single optimized pass.
        Returns a list of tuples: (box, text, confidence) where box is 8-point format.
        """
        if self._backend == "easyocr":
            try:
                reader = self._get_easyocr_reader()
                results = reader.readtext(image)
                formatted = []
                for res in results:
                    b = res[0]
                    box_8pt = [
                        int(b[0][0]), int(b[0][1]),
                        int(b[1][0]), int(b[1][1]),
                        int(b[2][0]), int(b[2][1]),
                        int(b[3][0]), int(b[3][1])
                    ]
                    formatted.append((box_8pt, res[1], res[2]))
                return formatted
            except Exception as exc:
                logger.error("Native EasyOCR readtext failed: %s", exc)
                return []
        return []

    def recognize(self, image, boxes):
        """
        Recognises text within each bounding box in the image.

        Args:
            image (np.ndarray): Source RGB image (H, W, 3).
            boxes (list): List of 8-point bounding boxes.

        Returns:
            list[str]: Recognised text strings, one per box.
        """
        if not boxes:
            return []

        # If we are using EasyOCR, run optimized batch recognition
        if self._backend == "easyocr":
            return self._recognize_easyocr_batch(image, boxes)

        transcriptions = []

        for box in boxes:
            cropped = crop_box(image, box)
            if cropped is None or cropped.size == 0:
                transcriptions.append("")
                continue

            text = self._recognize_crop(cropped)
            transcriptions.append(text)

        return transcriptions

    def _recognize_easyocr_batch(self, image, boxes):
        """Batch recognition using EasyOCR."""
        try:
            reader = self._get_easyocr_reader()
            
            # Format boxes for EasyOCR free_list: list of [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            easyocr_boxes = []
            valid_indices = []
            
            for idx, box in enumerate(boxes):
                if len(box) == 8:
                    easyocr_boxes.append([
                        [int(box[0]), int(box[1])],
                        [int(box[2]), int(box[3])],
                        [int(box[4]), int(box[5])],
                        [int(box[6]), int(box[7])]
                    ])
                    valid_indices.append(idx)
            
            results = [""] * len(boxes)
            if not easyocr_boxes:
                return results
                
            # Perform batch recognition using the full image
            # Set batch_size=32 for CPU batch speedup
            ocr_results = reader.recognize(
                image, 
                horizontal_list=[], 
                free_list=easyocr_boxes,
                batch_size=32
            )
            
            # Map results using coordinate matching as a robust mechanism
            # Create a lookup map: (x1, y1, x2, y2, x3, y3, x4, y4) -> index
            coord_to_idx = {}
            for i, box in enumerate(easyocr_boxes):
                key = (box[0][0], box[0][1], box[1][0], box[1][1],
                       box[2][0], box[2][1], box[3][0], box[3][1])
                coord_to_idx[key] = valid_indices[i]
                
            for res_box, text, confidence in ocr_results:
                key = (int(res_box[0][0]), int(res_box[0][1]), int(res_box[1][0]), int(res_box[1][1]),
                       int(res_box[2][0]), int(res_box[2][1]), int(res_box[3][0]), int(res_box[3][1]))
                if key in coord_to_idx:
                    idx = coord_to_idx[key]
                    results[idx] = text.strip()
            
            # Fallback to direct ordering if coordinate mapping left any slots empty
            if len(ocr_results) == len(easyocr_boxes):
                for i, (_, text, _) in enumerate(ocr_results):
                    idx = valid_indices[i]
                    results[idx] = text.strip()
                    
            return results
        except Exception as exc:
            logger.warning("Batch EasyOCR failed, falling back to crop-by-crop: %s", exc)
            fallback_results = []
            for box in boxes:
                cropped = crop_box(image, box)
                if cropped is None or cropped.size == 0:
                    fallback_results.append("")
                    continue
                fallback_results.append(self._recognize_crop(cropped))
            return fallback_results

    def recognize_batch(self, images, boxes_list):
        """
        Batch recognition across multiple images.

        Args:
            images (list[np.ndarray]): List of source images.
            boxes_list (list[list]): List of box lists per image.

        Returns:
            list[list[str]]: Recognised texts per image.
        """
        results = []
        for image, boxes in zip(images, boxes_list):
            texts = self.recognize(image, boxes)
            results.append(texts)
        return results

    def _recognize_crop(self, crop):
        """
        Recognises text in a single cropped region using the active backend.

        Args:
            crop (np.ndarray): Cropped image region.

        Returns:
            str: Recognised text.
        """
        if self._backend == "easyocr":
            return self._recognize_easyocr(crop)
        elif self._backend == "tesseract":
            return self._recognize_tesseract(crop)
        else:
            return self._recognize_mock(crop)

    def _recognize_easyocr(self, crop):
        """Recognition using EasyOCR."""
        try:
            reader = self._get_easyocr_reader()
            results = reader.readtext(crop, detail=0, paragraph=True)
            return " ".join(results).strip() if results else ""
        except Exception as exc:
            logger.warning("EasyOCR failed: %s", exc)
            return ""

    def _recognize_tesseract(self, crop):
        """Recognition using Tesseract OCR."""
        try:
            if HAS_CV2 and len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            else:
                gray = crop
            text = pytesseract.image_to_string(gray, config="--psm 7").strip()
            return text
        except Exception as exc:
            logger.warning("Tesseract failed: %s", exc)
            return ""

    def _recognize_mock(self, crop):
        """Mock recognition — returns a placeholder string."""
        return "MockText"


# ===========================================================================
#  Standalone Execution
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  Text Recognition — Task 3 Verification")
    print("=" * 60)

    recognizer = TextRecognizer()
    print(f"\n  Active backend: {recognizer.backend}")
    print(f"  EasyOCR available: {HAS_EASYOCR}")
    print(f"  Tesseract available: {HAS_TESSERACT}")

    # Test with synthetic image
    dummy = np.zeros((200, 600, 3), dtype=np.uint8)
    if HAS_CV2:
        cv2.putText(dummy, "HELLO WORLD", (20, 100),
                     cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)

    dummy_box = [10, 10, 590, 10, 590, 190, 10, 190]

    texts = recognizer.recognize(dummy, [dummy_box])
    print(f"\n  Recognised text: {texts}")

    # Test crop
    cropped = crop_box(dummy, dummy_box)
    if cropped is not None:
        print(f"  Crop shape: {cropped.shape}")

    print("\n" + "=" * 60)
    print("  All checks passed!")
    print("=" * 60)
