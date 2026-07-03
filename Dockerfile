# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# - tesseract-ocr for OCR recognition engine
# - libgl1-mesa-glx and libglib2.0-0 for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY src/ /app/src/
COPY scripts/ /app/scripts/

# Expose port 8000
EXPOSE 8000

# Command to run FastAPI server
CMD ["uvicorn", "src.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
