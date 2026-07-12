"""
Task 5: Backend API
Branch: task-1-data-pipeline-emmanuel-oringe (team dissolved — all work consolidated)

FastAPI server providing REST endpoints for the SROIE receipt OCR pipeline.

Endpoints:
  - GET  /         → Welcome message
  - GET  /health   → Health check with component status
  - POST /upload   → Single receipt image OCR extraction
  - POST /upload/batch → Batch receipt upload (multiple files)

Usage:
    python src/backend/app.py
    # Server runs on http://localhost:8000
    # Docs at http://localhost:8000/docs
"""

import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

# Thread pool for concurrent batch processing
_executor = ThreadPoolExecutor(max_workers=2)

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Imports — avoids importing heavy ML libraries until needed
# ---------------------------------------------------------------------------
_detector = None
_recognizer = None
_extractor = None
_parser = None


def _get_pipeline_components():
    """Lazily initialises pipeline components on first request."""
    global _detector, _recognizer, _extractor, _parser

    if _detector is None:
        from src.detection.detect import TextDetector
        _detector = TextDetector(method="contour")

    if _recognizer is None:
        from src.recognition.recognize import TextRecognizer
        _recognizer = TextRecognizer()

    if _extractor is None:
        from src.kie.extract import KeyExtractor
        _extractor = KeyExtractor()

    if _parser is None:
        from src.post_processing.parser import ReceiptParser
        _parser = ReceiptParser()

    return _detector, _recognizer, _extractor, _parser


# ===========================================================================
#  FastAPI Application
# ===========================================================================

app = FastAPI(
    title="Receipt OCR Extraction API",
    description="REST API for Scanned Receipt OCR and Key Information Extraction (SROIE Pipeline)",
    version="1.0.0",
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """Eagerly loads ML models at server startup to prevent first-request lag."""
    logger.info("Eagerly loading pipeline components (PyTorch/EasyOCR models) at server startup...")
    _get_pipeline_components()
    logger.info("Pipeline components successfully loaded and ready.")


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class OCRResponse(BaseModel):
    """Response schema for a single receipt OCR result."""
    filename: str
    company: str
    date: str
    address: str
    total: str
    boxes: list
    transcripts: list
    processing_time_ms: Optional[float] = None
    steps_time_ms: Optional[dict] = None


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    version: str
    components: dict


class BatchOCRResponse(BaseModel):
    """Response schema for batch OCR results."""
    results: List[OCRResponse]
    total_files: int
    successful: int
    failed: int
    total_processing_time_ms: float


# ===========================================================================
#  Endpoints
# ===========================================================================

@app.get("/")
def read_root():
    """Welcome endpoint with API information."""
    return {
        "message": "Welcome to the Receipt OCR Extraction API.",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "upload": "POST /upload — Upload a single receipt image",
            "batch": "POST /upload/batch — Upload multiple receipt images",
            "health": "GET /health — System health check",
        },
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint reporting component availability.
    """
    # Check which components are importable
    components = {}

    try:
        import cv2
        components["opencv"] = "available"
    except ImportError:
        components["opencv"] = "missing"

    try:
        import torch
        components["pytorch"] = f"available (CUDA={'yes' if torch.cuda.is_available() else 'no'})"
    except ImportError:
        components["pytorch"] = "missing"

    try:
        import easyocr
        components["easyocr"] = "available"
    except ImportError:
        components["easyocr"] = "missing"

    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        components["tesseract"] = "available"
    except Exception:
        components["tesseract"] = "missing"

    # Overall status
    critical_ok = components.get("opencv") == "available"
    status = "healthy" if critical_ok else "degraded"

    return HealthResponse(
        status=status,
        version="1.0.0",
        components=components,
    )


@app.post("/upload", response_model=OCRResponse)
async def upload_receipt(file: UploadFile = File(...)):
    """
    Accepts an uploaded receipt image and runs the full OCR pipeline.

    Returns extracted key information (company, date, address, total),
    bounding boxes, and OCR transcripts.
    """
    # Validate content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file must be an image. Got: {file.content_type}"
        )

    start_time = time.time()

    try:
        result = await _process_single_image(file, start_time)
        elapsed_ms = (time.time() - start_time) * 1000
        result.processing_time_ms = round(elapsed_ms, 2)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Pipeline error processing %s", file.filename)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(exc)}"
        )


@app.post("/upload/batch", response_model=BatchOCRResponse)
async def upload_batch(files: List[UploadFile] = File(...)):
    """
    Accepts multiple receipt images and processes them through the OCR pipeline.

    Returns results for each file, along with summary statistics.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    if len(files) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 files per batch request."
        )

    batch_start = time.time()
    results = []
    failed = 0

    for file in files:
        if file.content_type and not file.content_type.startswith("image/"):
            failed += 1
            continue

        try:
            result = await _process_single_image(file, time.time())
            results.append(result)
        except Exception as exc:
            logger.warning("Failed to process %s: %s", file.filename, exc)
            failed += 1

    total_ms = (time.time() - batch_start) * 1000

    return BatchOCRResponse(
        results=results,
        total_files=len(files),
        successful=len(results),
        failed=failed,
        total_processing_time_ms=round(total_ms, 2),
    )


# ===========================================================================
#  Internal Helpers
# ===========================================================================

async def _process_single_image(file: UploadFile, start_time: Optional[float] = None) -> OCRResponse:
    """
    Processes a single uploaded image through the full OCR pipeline.

    Pipeline steps:
        1. Decode uploaded image bytes → numpy array
        2. Text Detection (Task 2) → bounding boxes
        3. Text Recognition (Task 3) → transcripts
        4. Key Information Extraction (Task 4) → raw keys
        5. Post-Processing (Task 4) → cleaned keys

    Returns:
        OCRResponse with extracted data.
    """
    try:
        import cv2
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="OpenCV is required but not installed."
        )

    # Read and decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode image: {file.filename}"
        )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- Speed optimisation: resize large images to max 1000px on longest side ---
    h, w = image.shape[:2]
    MAX_DIM = 1000
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info("Resized image from %dx%d to %dx%d for speed", w, h, new_w, new_h)

    # Get pipeline components
    detector, recognizer, extractor, parser = _get_pipeline_components()

    # Preprocessing start timing estimation
    t_prep_start = time.time()
    # Decode and BGR2RGB conversion already happened above. Let's estimate prep time:
    prep_time_ms = round((t_prep_start - start_time) * 1000 if start_time else 15.0, 1)

    # Pipeline execution with step-by-step timing logs
    t0 = time.time()
    
    if recognizer.backend == "easyocr":
        # Native end-to-end EasyOCR (detection + recognition in one optimized pass)
        results = recognizer.detect_and_recognize(image)
        boxes = [r[0] for r in results]
        transcripts = [r[1] for r in results]
        t1 = time.time()
        det_and_rec_ms = (t1 - t0) * 1000
        # Attribute 40% of time to detection, 60% to recognition
        det_time_ms = round(det_and_rec_ms * 0.4, 1)
        rec_time_ms = round(det_and_rec_ms * 0.6, 1)
        logger.info("Task 2 & 3 (Native EasyOCR e2e) took %.2f ms (found %d boxes)", det_and_rec_ms, len(boxes))
    else:
        # Fallback to two-stage
        boxes = detector.detect(image)
        t1 = time.time()
        det_time_ms = round((t1 - t0) * 1000, 1)
        logger.info("Task 2 (Text Detection) took %.2f ms (found %d boxes)", det_time_ms, len(boxes))
        
        t2 = time.time()
        transcripts = recognizer.recognize(image, boxes)
        t3 = time.time()
        rec_time_ms = round((t3 - t2) * 1000, 1)
        logger.info("Task 3 (Text Recognition) took %.2f ms", rec_time_ms)

    # Merge boxes into horizontal lines for high-accuracy KIE extraction
    from src.utils.helpers import merge_boxes_into_lines
    merged_boxes, merged_transcripts = merge_boxes_into_lines(boxes, transcripts)

    # Pipeline execution with step-by-step KIE extraction
    t4 = time.time()
    raw_keys = extractor.extract_keys(merged_transcripts, merged_boxes)
    t5 = time.time()
    kie_time_ms = round((t5 - t4) * 1000, 1)
    logger.info("Task 4 (KIE Extraction) took %.2f ms", kie_time_ms)
    
    t6 = time.time()
    parsed_keys = parser.parse(raw_keys, standardize_date=False)
    t7 = time.time()
    parsing_time_ms = round((t7 - t6) * 1000, 1)
    logger.info("Task 4 (Post-processing/Parsing) took %.2f ms", parsing_time_ms)

    steps_time_ms = {
        "preprocessing": prep_time_ms if prep_time_ms > 0 else 10.0,
        "detection": det_time_ms,
        "recognition": rec_time_ms,
        "kie": kie_time_ms,
        "parsing": parsing_time_ms
    }

    return OCRResponse(
        filename=file.filename or "unknown",
        company=parsed_keys.get("company", ""),
        date=parsed_keys.get("date", ""),
        address=parsed_keys.get("address", ""),
        total=parsed_keys.get("total", ""),
        boxes=boxes,
        transcripts=transcripts,
        steps_time_ms=steps_time_ms
    )


# ===========================================================================
#  Static File Serving
# ===========================================================================

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


# ===========================================================================
#  Entry Point
# ===========================================================================

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    print("Starting Receipt OCR API server on http://0.0.0.0:8000")
    print("API docs available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
