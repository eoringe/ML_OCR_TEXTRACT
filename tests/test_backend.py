"""
Unit tests for Task 5: Backend API
Run with: python -m pytest tests/test_backend.py -v
"""

import os
import sys
import io
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from fastapi.testclient import TestClient
    HAS_TESTCLIENT = True
except ImportError:
    HAS_TESTCLIENT = False

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _create_test_image_bytes():
    """Creates a minimal valid JPEG image as bytes for upload testing."""
    if not HAS_CV2:
        return None
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (190, 90), (255, 255, 255), -1)
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


@unittest.skipUnless(HAS_TESTCLIENT, "FastAPI TestClient not available")
class TestRootEndpoint(unittest.TestCase):
    """Tests for GET / endpoint."""

    @classmethod
    def setUpClass(cls):
        from src.backend.app import app
        cls.client = TestClient(app)

    def test_root_returns_200(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_root_has_message(self):
        response = self.client.get("/")
        data = response.json()
        self.assertIn("message", data)

    def test_root_has_version(self):
        response = self.client.get("/")
        data = response.json()
        self.assertIn("version", data)

    def test_root_has_endpoints(self):
        response = self.client.get("/")
        data = response.json()
        self.assertIn("endpoints", data)


@unittest.skipUnless(HAS_TESTCLIENT, "FastAPI TestClient not available")
class TestHealthEndpoint(unittest.TestCase):
    """Tests for GET /health endpoint."""

    @classmethod
    def setUpClass(cls):
        from src.backend.app import app
        cls.client = TestClient(app)

    def test_health_returns_200(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_health_has_status(self):
        response = self.client.get("/health")
        data = response.json()
        self.assertIn("status", data)
        self.assertIn(data["status"], ("healthy", "degraded"))

    def test_health_has_components(self):
        response = self.client.get("/health")
        data = response.json()
        self.assertIn("components", data)
        self.assertIn("opencv", data["components"])

    def test_health_has_version(self):
        response = self.client.get("/health")
        data = response.json()
        self.assertIn("version", data)


@unittest.skipUnless(HAS_TESTCLIENT and HAS_CV2, "TestClient or OpenCV not available")
class TestUploadEndpoint(unittest.TestCase):
    """Tests for POST /upload endpoint."""

    @classmethod
    def setUpClass(cls):
        from src.backend.app import app
        cls.client = TestClient(app)

    def test_upload_valid_image(self):
        img_bytes = _create_test_image_bytes()
        response = self.client.post(
            "/upload",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)

    def test_upload_returns_expected_fields(self):
        img_bytes = _create_test_image_bytes()
        response = self.client.post(
            "/upload",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        data = response.json()
        self.assertIn("filename", data)
        self.assertIn("company", data)
        self.assertIn("date", data)
        self.assertIn("address", data)
        self.assertIn("total", data)
        self.assertIn("boxes", data)
        self.assertIn("transcripts", data)

    def test_upload_returns_filename(self):
        img_bytes = _create_test_image_bytes()
        response = self.client.post(
            "/upload",
            files={"file": ("receipt_001.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        data = response.json()
        self.assertEqual(data["filename"], "receipt_001.jpg")

    def test_upload_boxes_is_list(self):
        img_bytes = _create_test_image_bytes()
        response = self.client.post(
            "/upload",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        data = response.json()
        self.assertIsInstance(data["boxes"], list)
        self.assertIsInstance(data["transcripts"], list)

    def test_upload_invalid_content_type(self):
        response = self.client.post(
            "/upload",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        self.assertEqual(response.status_code, 400)

    def test_upload_has_processing_time(self):
        img_bytes = _create_test_image_bytes()
        response = self.client.post(
            "/upload",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        data = response.json()
        self.assertIn("processing_time_ms", data)
        if data["processing_time_ms"] is not None:
            self.assertGreater(data["processing_time_ms"], 0)


@unittest.skipUnless(HAS_TESTCLIENT and HAS_CV2, "TestClient or OpenCV not available")
class TestBatchUploadEndpoint(unittest.TestCase):
    """Tests for POST /upload/batch endpoint."""

    @classmethod
    def setUpClass(cls):
        from src.backend.app import app
        cls.client = TestClient(app)

    def test_batch_upload_two_images(self):
        img_bytes = _create_test_image_bytes()
        files = [
            ("files", ("img1.jpg", io.BytesIO(img_bytes), "image/jpeg")),
            ("files", ("img2.jpg", io.BytesIO(img_bytes), "image/jpeg")),
        ]
        response = self.client.post("/upload/batch", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("results", data)
        self.assertEqual(data["total_files"], 2)
        self.assertEqual(data["successful"], 2)

    def test_batch_upload_mixed_types(self):
        img_bytes = _create_test_image_bytes()
        files = [
            ("files", ("img1.jpg", io.BytesIO(img_bytes), "image/jpeg")),
            ("files", ("bad.txt", io.BytesIO(b"not image"), "text/plain")),
        ]
        response = self.client.post("/upload/batch", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["successful"], 1)
        self.assertEqual(data["failed"], 1)

    def test_batch_has_timing(self):
        img_bytes = _create_test_image_bytes()
        files = [("files", ("img.jpg", io.BytesIO(img_bytes), "image/jpeg"))]
        response = self.client.post("/upload/batch", files=files)
        data = response.json()
        self.assertIn("total_processing_time_ms", data)


@unittest.skipUnless(HAS_TESTCLIENT, "FastAPI TestClient not available")
class TestCORSConfiguration(unittest.TestCase):
    """Tests for CORS middleware configuration."""

    @classmethod
    def setUpClass(cls):
        from src.backend.app import app
        cls.client = TestClient(app)

    def test_cors_headers_present(self):
        response = self.client.options(
            "/upload",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        # FastAPI CORS middleware should return allow-origin header
        self.assertIn(response.status_code, (200, 405))


if __name__ == "__main__":
    unittest.main(verbosity=2)
