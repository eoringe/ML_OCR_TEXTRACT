import os
import sys
import time
import logging
import cv2
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_pipeline.prep import preprocess_image
from src.detection.detect import TextDetector
from src.recognition.recognize import TextRecognizer
from src.kie.extract import KeyExtractor
from src.post_processing.parser import ReceiptParser

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("receipt_ocr_api")

# Constants
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB per README specification
MAX_BATCH_FILES = 10  # Maximum images in a single batch request
DATASET_DIR = os.path.join(project_root, "dataset")

# Pydantic Response Models
class HealthResponse(BaseModel):
    status: str
    version: str


class OCRResponse(BaseModel):
    filename: str
    company: str
    date: str
    address: str
    total: str
    boxes: list
    transcripts: list
    image_width: int
    image_height: int
    processing_time_ms: float


class BatchOCRResponse(BaseModel):
    results: List[OCRResponse]
    total_processing_time_ms: float
    file_count: int


class SampleListResponse(BaseModel):
    samples: List[str]
    count: int

# FastAPI Application
app = FastAPI(
    title="Receipt OCR Extraction API",
    description="API for Scanned Receipts OCR and Key Information Extraction (SROIE pipeline)",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline Component Initialization
detector = TextDetector()
recognizer = TextRecognizer()
extractor = KeyExtractor()
parser = ReceiptParser()

logger.info("Pipeline components initialized: TextDetector, TextRecognizer, KeyExtractor, ReceiptParser")

# Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Return structured JSON for all HTTP errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": True, "detail": exc.detail, "status_code": exc.status_code},
    )

# Shared Helper: run the full OCR pipeline on an in-memory image

def _run_pipeline(image: np.ndarray) -> dict:
    """
    Execute the full detection → recognition → KIE → parsing pipeline on an
    RGB numpy image and return the extracted result dictionary.
    """
    # 1. Text Detection (Task 2)
    boxes = detector.detect(image)

    # 2. Text Recognition (Task 3)
    transcripts = recognizer.recognize(image, boxes)

    # 3. Key Information Extraction (Task 4)
    raw_keys = extractor.extract_keys(transcripts, boxes)

    # 4. Post-Processing & Parsing
    parsed_keys = parser.parse(raw_keys)

    return {
        "boxes": boxes,
        "transcripts": transcripts,
        "parsed_keys": parsed_keys,
    }


def _decode_image_bytes(contents: bytes) -> np.ndarray:
    """
    Decode raw file bytes into an RGB numpy image.
    Raises ValueError if decoding fails.
    """
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image — the file may be corrupt or not a valid image format.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Endpoints

@app.get("/", tags=["General"])
def read_root():
    """Root endpoint with welcome message and usage hints."""
    return {
        "message": "Welcome to the Receipt OCR Extraction API. Use /docs for documentation, or upload files at /upload",
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Health-check / readiness probe for Docker and frontend connection checks."""
    return HealthResponse(status="ok", version="1.0.0")

@app.post("/upload", response_model=OCRResponse, tags=["OCR Pipeline"])
async def upload_receipt(file: UploadFile = File(...)):
    """
    Accepts a single uploaded receipt image, runs the full OCR pipeline, and
    returns extracted key information along with bounding boxes and transcripts.
    """
    # --- Validate content type ---
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (e.g. image/jpeg, image/png).")

    # --- Read and validate size ---
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the maximum allowed size of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    try:
        # Decode image
        image = _decode_image_bytes(contents)
        h, w = image.shape[:2]

        # Run the pipeline with timing
        start_time = time.time()
        result = _run_pipeline(image)
        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(
            "Processed '%s' (%dx%d) in %.1f ms — %d boxes detected",
            file.filename, w, h, elapsed_ms, len(result["boxes"]),
        )

        return OCRResponse(
            filename=file.filename or "unknown",
            company=result["parsed_keys"].get("company", ""),
            date=result["parsed_keys"].get("date", ""),
            address=result["parsed_keys"].get("address", ""),
            total=result["parsed_keys"].get("total", ""),
            boxes=result["boxes"],
            transcripts=result["transcripts"],
            image_width=w,
            image_height=h,
            processing_time_ms=round(elapsed_ms, 2),
        )

    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        logger.exception("Pipeline error while processing '%s'", file.filename)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.post("/batch", response_model=BatchOCRResponse, tags=["OCR Pipeline"])
async def batch_upload_receipts(files: List[UploadFile] = File(...)):
    """
    Accepts up to 10 receipt images and returns OCR results for each.
    Useful for batch evaluation (Task 7) and bulk processing.
    """
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Batch upload limited to {MAX_BATCH_FILES} files. Received {len(files)}.",
        )

    results: List[OCRResponse] = []
    batch_start = time.time()

    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            logger.warning("Skipping non-image file in batch: '%s' (type: %s)", file.filename, file.content_type)
            continue

        contents = await file.read()
        if len(contents) > MAX_UPLOAD_SIZE_BYTES:
            logger.warning("Skipping oversized file in batch: '%s' (%d bytes)", file.filename, len(contents))
            continue

        try:
            image = _decode_image_bytes(contents)
            h, w = image.shape[:2]

            start_time = time.time()
            result = _run_pipeline(image)
            elapsed_ms = (time.time() - start_time) * 1000

            results.append(OCRResponse(
                filename=file.filename or "unknown",
                company=result["parsed_keys"].get("company", ""),
                date=result["parsed_keys"].get("date", ""),
                address=result["parsed_keys"].get("address", ""),
                total=result["parsed_keys"].get("total", ""),
                boxes=result["boxes"],
                transcripts=result["transcripts"],
                image_width=w,
                image_height=h,
                processing_time_ms=round(elapsed_ms, 2),
            ))
        except Exception as e:
            logger.error("Failed to process '%s' in batch: %s", file.filename, str(e))

    total_elapsed_ms = (time.time() - batch_start) * 1000
    logger.info("Batch complete: %d/%d files processed in %.1f ms", len(results), len(files), total_elapsed_ms)

    return BatchOCRResponse(
        results=results,
        total_processing_time_ms=round(total_elapsed_ms, 2),
        file_count=len(results),
    )


@app.get("/samples", response_model=SampleListResponse, tags=["Dataset"])
def list_samples():
    """
    List available sample receipt filenames from the local SROIE dataset.
    Returns an empty list if the dataset has not been downloaded yet.
    """
    img_dir = os.path.join(DATASET_DIR, "img")
    if not os.path.isdir(img_dir):
        logger.info("Dataset image directory not found at %s — returning empty sample list", img_dir)
        return SampleListResponse(samples=[], count=0)

    image_extensions = {".jpg", ".jpeg", ".png"}
    samples = sorted(
        f for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    )
    return SampleListResponse(samples=samples, count=len(samples))


@app.get("/sample/{filename}", tags=["Dataset"])
def get_sample(filename: str):
    """
    Serve a single sample receipt image from the local SROIE dataset.
    Returns 404 if the dataset is missing or the file does not exist.
    """
    img_dir = os.path.join(DATASET_DIR, "img")
    file_path = os.path.join(img_dir, filename)

    # Security: prevent path traversal
    resolved = os.path.realpath(file_path)
    if not resolved.startswith(os.path.realpath(img_dir)):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not os.path.isfile(resolved):
        raise HTTPException(status_code=404, detail=f"Sample image '{filename}' not found.")

    return FileResponse(resolved, media_type="image/jpeg")


# Static File Serving (Frontend)
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# Main Entry Point
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Receipt OCR API server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
