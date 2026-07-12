# ==============================================================
# SROIE Receipt OCR Pipeline — Docker Image
# ==============================================================
# Multi-stage build for a smaller final image.
# Stage 1: Install Python dependencies.
# Stage 2: Copy source code and run the FastAPI server.

# --- Stage 1: Build dependencies ---
FROM python:3.10-slim AS builder

WORKDIR /build

COPY requirements.txt /build/
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.10-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

# Install system dependencies for OpenCV and Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy project source code
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY requirements.txt /app/
COPY README.md /app/

# Create dataset directory placeholder (dataset should be mounted at runtime)
RUN mkdir -p /app/dataset /app/checkpoints

# Expose the API port
EXPOSE 8000

# Health check — verify the server is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI server
CMD ["uvicorn", "src.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
