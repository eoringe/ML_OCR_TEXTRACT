"""
Comprehensive test suite for the Receipt OCR Backend API (Task 5).

Tests cover:
- Root & health endpoints
- Single image upload (happy path, validation, corrupt data)
- Batch upload endpoint
- Dataset sample browser endpoints
- CORS header presence
- Response schema validation

All tests use synthetic images — no SROIE dataset download is required.
"""

import io
import json
import pytest


# ============================================================================
# General Endpoints
# ============================================================================

class TestRootEndpoint:
    """Tests for GET /"""

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_welcome_message(self, client):
        data = client.get("/").json()
        assert "message" in data
        assert "Welcome" in data["message"]


class TestHealthEndpoint:
    """Tests for GET /health"""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_includes_version(self, client):
        data = client.get("/health").json()
        assert "version" in data
        assert isinstance(data["version"], str)


# ============================================================================
# Single Upload Endpoint
# ============================================================================

class TestUploadEndpoint:
    """Tests for POST /upload"""

    def test_upload_valid_image_returns_200(self, client, synthetic_receipt_jpeg):
        response = client.post(
            "/upload",
            files={"file": ("receipt.jpg", io.BytesIO(synthetic_receipt_jpeg), "image/jpeg")},
        )
        assert response.status_code == 200

    def test_upload_response_has_required_fields(self, client, synthetic_receipt_jpeg):
        response = client.post(
            "/upload",
            files={"file": ("receipt.jpg", io.BytesIO(synthetic_receipt_jpeg), "image/jpeg")},
        )
        data = response.json()

        required_fields = [
            "filename", "company", "date", "address", "total",
            "boxes", "transcripts", "image_width", "image_height",
            "processing_time_ms",
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_upload_filename_matches(self, client, synthetic_receipt_jpeg):
        response = client.post(
            "/upload",
            files={"file": ("my_receipt.jpg", io.BytesIO(synthetic_receipt_jpeg), "image/jpeg")},
        )
        assert response.json()["filename"] == "my_receipt.jpg"

    def test_upload_image_dimensions_correct(self, client, synthetic_receipt_jpeg):
        response = client.post(
            "/upload",
            files={"file": ("receipt.jpg", io.BytesIO(synthetic_receipt_jpeg), "image/jpeg")},
        )
        data = response.json()
        # Our synthetic image is 300x400
        assert data["image_width"] == 300
        assert data["image_height"] == 400

    def test_upload_processing_time_positive(self, client, synthetic_receipt_jpeg):
        response = client.post(
            "/upload",
            files={"file": ("receipt.jpg", io.BytesIO(synthetic_receipt_jpeg), "image/jpeg")},
        )
        assert response.json()["processing_time_ms"] >= 0

    def test_upload_boxes_is_list(self, client, synthetic_receipt_jpeg):
        response = client.post(
            "/upload",
            files={"file": ("receipt.jpg", io.BytesIO(synthetic_receipt_jpeg), "image/jpeg")},
        )
        data = response.json()
        assert isinstance(data["boxes"], list)

    def test_upload_transcripts_is_list(self, client, synthetic_receipt_jpeg):
        response = client.post(
            "/upload",
            files={"file": ("receipt.jpg", io.BytesIO(synthetic_receipt_jpeg), "image/jpeg")},
        )
        data = response.json()
        assert isinstance(data["transcripts"], list)

    def test_upload_boxes_and_transcripts_same_length(self, client, synthetic_receipt_jpeg):
        response = client.post(
            "/upload",
            files={"file": ("receipt.jpg", io.BytesIO(synthetic_receipt_jpeg), "image/jpeg")},
        )
        data = response.json()
        assert len(data["boxes"]) == len(data["transcripts"])


class TestUploadValidation:
    """Tests for input validation on POST /upload"""

    def test_upload_non_image_rejected(self, client, synthetic_text_file):
        response = client.post(
            "/upload",
            files={"file": ("notes.txt", io.BytesIO(synthetic_text_file), "text/plain")},
        )
        assert response.status_code == 400

    def test_upload_non_image_error_message(self, client, synthetic_text_file):
        response = client.post(
            "/upload",
            files={"file": ("notes.txt", io.BytesIO(synthetic_text_file), "text/plain")},
        )
        data = response.json()
        assert "detail" in data
        assert "image" in data["detail"].lower()

    def test_upload_oversized_file_rejected(self, client, oversized_payload):
        response = client.post(
            "/upload",
            files={"file": ("huge.jpg", io.BytesIO(oversized_payload), "image/jpeg")},
        )
        assert response.status_code == 413

    def test_upload_corrupt_image_returns_500(self, client):
        """Send random bytes claiming to be a JPEG — should fail at decode."""
        corrupt_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # JPEG SOI marker but junk data
        response = client.post(
            "/upload",
            files={"file": ("corrupt.jpg", io.BytesIO(corrupt_bytes), "image/jpeg")},
        )
        assert response.status_code == 500

    def test_upload_png_accepted(self, client):
        """Verify that PNG images (not just JPEG) are accepted."""
        import cv2
        import numpy as np

        img = np.ones((100, 100, 3), dtype=np.uint8) * 200
        _, buf = cv2.imencode(".png", img)
        png_bytes = buf.tobytes()

        response = client.post(
            "/upload",
            files={"file": ("receipt.png", io.BytesIO(png_bytes), "image/png")},
        )
        assert response.status_code == 200


# ============================================================================
# Batch Upload Endpoint
# ============================================================================

class TestBatchUpload:
    """Tests for POST /batch"""

    def test_batch_upload_two_images(self, client, synthetic_receipt_jpeg):
        files = [
            ("files", ("receipt1.jpg", io.BytesIO(synthetic_receipt_jpeg), "image/jpeg")),
            ("files", ("receipt2.jpg", io.BytesIO(synthetic_receipt_jpeg), "image/jpeg")),
        ]
        response = client.post("/batch", files=files)
        assert response.status_code == 200

        data = response.json()
        assert data["file_count"] == 2
        assert len(data["results"]) == 2

    def test_batch_upload_response_schema(self, client, synthetic_receipt_jpeg):
        files = [
            ("files", ("receipt.jpg", io.BytesIO(synthetic_receipt_jpeg), "image/jpeg")),
        ]
        response = client.post("/batch", files=files)
        data = response.json()

        assert "results" in data
        assert "total_processing_time_ms" in data
        assert "file_count" in data
        assert data["total_processing_time_ms"] >= 0

    def test_batch_upload_skips_non_images(self, client, synthetic_receipt_jpeg, synthetic_text_file):
        files = [
            ("files", ("receipt.jpg", io.BytesIO(synthetic_receipt_jpeg), "image/jpeg")),
            ("files", ("notes.txt", io.BytesIO(synthetic_text_file), "text/plain")),
        ]
        response = client.post("/batch", files=files)
        data = response.json()

        # Only the image should be processed; the text file should be skipped
        assert data["file_count"] == 1

    def test_batch_upload_exceeds_limit(self, client, synthetic_receipt_jpeg):
        """Sending more than 10 files should be rejected."""
        files = [
            ("files", (f"receipt_{i}.jpg", io.BytesIO(synthetic_receipt_jpeg), "image/jpeg"))
            for i in range(11)
        ]
        response = client.post("/batch", files=files)
        assert response.status_code == 400

    def test_batch_each_result_has_filename(self, client, synthetic_receipt_jpeg):
        files = [
            ("files", ("alpha.jpg", io.BytesIO(synthetic_receipt_jpeg), "image/jpeg")),
            ("files", ("beta.jpg", io.BytesIO(synthetic_receipt_jpeg), "image/jpeg")),
        ]
        response = client.post("/batch", files=files)
        data = response.json()

        filenames = [r["filename"] for r in data["results"]]
        assert "alpha.jpg" in filenames
        assert "beta.jpg" in filenames


# ============================================================================
# Sample Browser Endpoints
# ============================================================================

class TestSamplesEndpoints:
    """Tests for GET /samples and GET /sample/{filename}"""

    def test_samples_list_returns_200(self, client):
        response = client.get("/samples")
        assert response.status_code == 200

    def test_samples_list_schema(self, client):
        data = client.get("/samples").json()
        assert "samples" in data
        assert "count" in data
        assert isinstance(data["samples"], list)
        assert isinstance(data["count"], int)
        assert data["count"] == len(data["samples"])

    def test_sample_nonexistent_file_returns_404(self, client):
        response = client.get("/sample/nonexistent_file_xyz.jpg")
        assert response.status_code == 404

    def test_sample_path_traversal_blocked(self, client):
        """Ensure directory traversal attacks are rejected."""
        response = client.get("/sample/../../etc/passwd")
        assert response.status_code in (400, 404)


# ============================================================================
# CORS Headers
# ============================================================================

class TestCORSHeaders:
    """Verify CORS middleware is configured correctly."""

    def test_cors_allows_any_origin(self, client):
        response = client.options(
            "/upload",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.headers.get("access-control-allow-origin") in ("*", "http://example.com")

    def test_cors_on_get_endpoint(self, client):
        response = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert "access-control-allow-origin" in response.headers


# ============================================================================
# OCR Response Schema Deep Validation
# ============================================================================

class TestOCRResponseSchema:
    """Validate the complete structure and types of the OCR response."""

    def test_all_string_fields_are_strings(self, client, synthetic_receipt_jpeg):
        response = client.post(
            "/upload",
            files={"file": ("receipt.jpg", io.BytesIO(synthetic_receipt_jpeg), "image/jpeg")},
        )
        data = response.json()

        for field in ["filename", "company", "date", "address", "total"]:
            assert isinstance(data[field], str), f"Field '{field}' should be a string, got {type(data[field])}"

    def test_numeric_fields_are_numbers(self, client, synthetic_receipt_jpeg):
        response = client.post(
            "/upload",
            files={"file": ("receipt.jpg", io.BytesIO(synthetic_receipt_jpeg), "image/jpeg")},
        )
        data = response.json()

        assert isinstance(data["image_width"], int)
        assert isinstance(data["image_height"], int)
        assert isinstance(data["processing_time_ms"], (int, float))

    def test_boxes_contain_8_coordinates_each(self, client, synthetic_receipt_jpeg):
        response = client.post(
            "/upload",
            files={"file": ("receipt.jpg", io.BytesIO(synthetic_receipt_jpeg), "image/jpeg")},
        )
        data = response.json()

        for i, box in enumerate(data["boxes"]):
            assert isinstance(box, list), f"Box {i} should be a list"
            assert len(box) == 8, f"Box {i} should have 8 coordinates, got {len(box)}"

    def test_response_serializable_as_json(self, client, synthetic_receipt_jpeg):
        """Ensure the response can be round-tripped through JSON."""
        response = client.post(
            "/upload",
            files={"file": ("receipt.jpg", io.BytesIO(synthetic_receipt_jpeg), "image/jpeg")},
        )
        raw = response.text
        # Should not raise
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)
