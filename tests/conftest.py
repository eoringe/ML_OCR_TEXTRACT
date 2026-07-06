"""
Shared pytest fixtures for the Receipt OCR API test suite.

Provides reusable fixtures for:
- FastAPI TestClient
- Synthetic JPEG receipt images (no dataset dependency)
- Non-image file payloads for rejection testing
"""

import io
import os
import sys
import pytest
import cv2
import numpy as np

# Ensure project root is on the path so `src.*` imports resolve
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi.testclient import TestClient
from src.backend.app import app


@pytest.fixture(scope="module")
def client():
    """
    Module-scoped TestClient instance.
    Reused across all tests in a module to avoid reinitialising the app
    (and therefore the heavy ML pipeline objects) on every test.
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def synthetic_receipt_jpeg() -> bytes:
    """
    Creates a minimal in-memory JPEG image that resembles a receipt.
    Draws some text lines so the contour-based detector has something to find.
    No dependency on the SROIE dataset.
    """
    # White canvas simulating a receipt
    img = np.ones((400, 300, 3), dtype=np.uint8) * 255

    # Draw a few text-like lines in black
    cv2.putText(img, "BOOK STORE INC", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "123 Main Street", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Date: 12/25/2023", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Item A    $10.00", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "TOTAL     $10.80", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Encode to JPEG bytes
    success, buf = cv2.imencode(".jpg", img)
    assert success, "Failed to encode synthetic JPEG"
    return buf.tobytes()


@pytest.fixture()
def synthetic_text_file() -> bytes:
    """
    Creates a plain-text payload to verify the API rejects non-image uploads.
    """
    return b"This is not an image file. Just plain text content."


@pytest.fixture()
def oversized_payload() -> bytes:
    """
    Creates a payload that exceeds the 10 MB upload limit.
    Uses random bytes rather than a real image to keep it lightweight.
    """
    return b"\x00" * (11 * 1024 * 1024)  # 11 MB of null bytes
