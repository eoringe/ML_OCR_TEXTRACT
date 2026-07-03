import os
import sys
import shutil
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_pipeline.prep import preprocess_image
from src.detection.detect import TextDetector
from src.recognition.recognize import TextRecognizer
from src.kie.extract import KeyExtractor
from src.post_processing.parser import ReceiptParser

# Initialize FastAPI App
app = FastAPI(
    title="Receipt OCR Extraction API",
    description="API for Scanned Receipt OCR and Key Information Extraction",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipelines
detector = TextDetector()
recognizer = TextRecognizer()
extractor = KeyExtractor()
parser = ReceiptParser()

class OCRResponse(BaseModel):
    filename: str
    company: str
    date: str
    address: str
    total: str
    boxes: list
    transcripts: list

@app.get("/")
def read_root():
    return {"message": "Welcome to the Receipt OCR Extraction API. Use /docs for documentation, or upload files at /upload"}

@app.post("/upload", response_model=OCRResponse)
async def upload_receipt(file: UploadFile = File(...)):
    """
    Accepts an uploaded receipt image, runs the full OCR pipeline, and returns KIE details.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        
    try:
        # Read uploaded image bytes
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. Image Preprocessing (Task 1)
        # Note: Preprocessing happens inside detectors or as a standalone step.
        # We pass original RGB image to the detector/recognizer.
        
        # 2. Text Detection (Task 2)
        boxes = detector.detect(image)
        
        # 3. Text Recognition (Task 3)
        transcripts = recognizer.recognize(image, boxes)
        
        # 4. Key Information Extraction (Task 4)
        raw_keys = extractor.extract_keys(transcripts, boxes)
        
        # 5. Post-Processing & Parsing (Task 5)
        parsed_keys = parser.parse(raw_keys)
        
        # Return response matching OCRResponse schema
        return OCRResponse(
            filename=file.filename,
            company=parsed_keys.get("company", ""),
            date=parsed_keys.get("date", ""),
            address=parsed_keys.get("address", ""),
            total=parsed_keys.get("total", ""),
            boxes=boxes,
            transcripts=transcripts
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

# Serve frontend static files if directory exists
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
