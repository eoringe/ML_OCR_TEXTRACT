# Scanned Receipts OCR & Information Extraction (SROIE) Pipeline

Welcome to the **ML OCR Receipt Text Extraction** project repository! This repository is configured for a collaborative development sprint by **7 team members** targeting full delivery in **1 week**.

The project uses the benchmark **SROIE (Scanned Receipts OCR and Information Extraction)** dataset to localise bounding boxes, recognize raw text, and extract key semantic fields: **Company, Date, Address, and Total**.

---

## 🚀 Getting Started

Follow these steps to set up your local development environment.

### 1. Python Environment Setup
We recommend using Python 3.10+ and creating a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download and Extract SROIE Dataset
We have created an automated dataset downloader script. Run it from the repository root to download and set up the dataset locally:
```bash
python scripts/download_dataset.py
```
This script will download the SROIE ZIP file from GitHub, extract the data files directly to `dataset/` (divided into `img/`, `box/`, and `key/`), and clean up all temporary files.
*Note: The `dataset/` directory is listed in `.gitignore` to avoid bloating the Git history.*

### 3. Run the Backend API
Start the FastAPI server:
```bash
python src/backend/app.py
```
The server runs on `http://localhost:8000`. You can visit `http://localhost:8000/docs` for API documentation.

### 4. Open the Web Frontend
You can open the user interface by launching `src/frontend/index.html` in your web browser. Drag and drop any receipt image from `dataset/img/` to inspect text detection bounding boxes and see extracted structured key fields!

---

## 📁 Repository Directory Structure

```text
ML_OCR_TEXTRACT/
├── dataset/                     # Downloaded receipt images and labels (Git ignored)
│   ├── img/                     # 1,000 scanned receipt JPEG files
│   ├── box/                     # Bounding boxes + transcript annotations (.txt)
│   └── key/                     # Target key extraction labels (.json)
├── scripts/
│   ├── download_dataset.py      # Dataset fetching script
│   └── setup_branches.py        # Automated team branch setup script
├── src/
│   ├── data_pipeline/
│   │   └── prep.py              # Task 1: Image loading, transforms, and splits
│   ├── detection/
│   │   └── detect.py            # Task 2: Text detection model wrapper
│   ├── recognition/
│   │   └── recognize.py         # Task 3: Text recognition CRNN wrapper
│   ├── kie/
│   │   └── extract.py           # Task 4: Key Information Extraction (LayoutLM)
│   ├── post_processing/
│   │   └── parser.py            # Task 4: OCR error corrections & regex cleaning
│   ├── backend/
│   │   └── app.py               # Task 5: FastAPI server and upload endpoint
│   ├── frontend/                # Task 6: UI Dashboard (Single-page app)
│   │   ├── index.html
│   │   ├── styles.css
│   │   └── app.js
│   └── utils/
│       ├── helpers.py           # Geometry, canvas overlays, and directories
│       └── evaluate.py          # Task 7: CER, WER, and KIE validation metrics
├── Dockerfile                   # Deployment container blueprint (Task 7)
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation (this file)
```

---

## 👥 7-Member Task & Role Allocation

Each team member is assigned to one of the 7 roles making up the extraction pipeline. **No work should ever be committed directly to the `main` branch.**

Here is the work mapping and starting branch for each team member:

| Member Name | Assigned Role | Core Responsibilities | Main Code Files | Starting Branch |
| :--- | :--- | :--- | :--- | :--- |
| **Emmanuel Oringe** | Task 1: Data Preprocessing | Image resizing, thresholding, augmentations, and PyTorch dataset splits. | [prep.py](src/data_pipeline/prep.py) | `task-1-data-pipeline-emmanuel-oringe` |
| **Peter Tobiko** | Task 2: Text Detection | Developing and optimizing text localization (e.g. OpenCV Contours, DBNet, or YOLOv8-OBB). | [detect.py](src/detection/detect.py) | `task-2-text-detection-peter-tobiko` |
| **Joy Chepchumba** | Task 3: Text Recognition | Implementing character transcription from bounding boxes (e.g. CRNN + CTC Loss, EasyOCR). | [recognize.py](src/recognition/recognize.py) | `task-3-text-recognition-joy-chepchumba` |
| **Kristein Mwaura** | Task 4: KIE & Post-Processing | Entity classification (LayoutLM/NLP) & character error corrections. | [extract.py](src/kie/extract.py), [parser.py](src/post_processing/parser.py) | `task-4-kie-parsing-kristein-mwaura` |
| **Victor Kimotho** | Task 5: Backend API | Designing and optimizing the FastAPI server endpoints. | [app.py](src/backend/app.py) | `task-5-backend-api-victor-kimotho` |
| **Grace Macharia** | Task 6: Frontend UI Dashboard | Interactive receipt visualizer, canvas overlays, coordinate scaling, and JSON inspector. | [frontend/](src/frontend/) | `task-6-frontend-ui-grace-macharia` |
| **Billiart Muthoni** | Task 7: MLOps, Eval & Docker | Metrics calculations (CER, WER, F1-scores), writing Dockerfiles, and CI configuration. | [evaluate.py](src/utils/evaluate.py), [Dockerfile](Dockerfile) | `task-7-mlops-eval-billiart-muthoni` |

---

## 📅 1-Week Project Roadmap

### Day 1: Project Alignment & Environment
* Set up virtual environments, install dependencies, and download the dataset (`download_dataset.py`).
* Verify directory skeletons locally. Ensure branches are checked out properly.

### Day 2: Baseline Pipeline Execution
* **Task 1-4**: Connect the heuristic/OpenCV contours and regex baselines.
* **Task 5-6**: Establish a connection between the Frontend and FastAPI upload endpoint.
* **Task 7**: Compute initial pipeline accuracy score (baseline).

### Day 3: Model Architecture & Prototyping
* **Task 1**: Refine data augmentation pipeline (rotation, scaling, contrast adjustment).
* **Task 2-4**: Replace baseline heuristics with neural networks (e.g., PyTorch CRNN for recognition, LayoutLM for KIE).
* Run initial trainings on the SROIE training set split.

### Day 4: Deep Learning Pipeline Training
* Complete model training and save checkpoints inside `checkpoints/`.
* Optimize learning rates and hyperparameters.

### Day 5: Pipeline Integration & Refinements
* Hook model weights into the FastAPI server.
* Perform end-to-end local testing. Debug API response speed and coordinate scaling in the frontend canvas overlay.

### Day 6: Validation, MLOps, & Deployment
* **Task 7**: Run final evaluation over test split, generating CER, WER, and KIE F1-scores.
* Package the application inside the Docker container. Confirm `docker build -t ocr-textract .` works.

### Day 7: final Demonstrations & Merges
* Merge all branches into `main` using Pull Requests. Resolve merge conflicts.
* Present the fully integrated OCR dashboard!

---

## 🛠 Git Workflow Rules

1. **Check out your task branch**:
   ```bash
   git checkout <your-starting-branch>
   ```
2. **Commit frequently**: Use clear commit messages descriptive of your task (e.g., `feat(detect): add contour text clustering baseline`).
3. **Pull Request Policy**: Submit a Pull Request to merge your branch into `main`. At least one review is required from an adjacent pipeline layer.
