# Traffic Rule Violation Detection — AID 728 Final Submission

This repository contains the final computer vision violation detection system for **Group 28**.

## 📂 Main Submission Folder

All submission files, report document, source code, and local model weights are organized under the **[Group28](./Group28)** directory.

Please refer to **[Group28/README.md](./Group28/README.md)** for detailed installation, dependencies setup, and usage instructions.

## 🎯 Quick Overview

Inside the submission folder, we provide two distinct implementations of the violation detector:
1. **PaddleOCR PP-OCRv5 Pipeline (`Group28/solution.py`)**: Uses custom YOLO models and offline PaddleOCR for maximum speed and accuracy.
2. **6-Stage EasyOCR Pipeline (`Group28/solution_6stage.py`)**: Incorporates a 6-stage image preprocessing pipeline with Test-Time Augmentation (TTA) and offline EasyOCR.
3. **Project Report (`Group28/Group28.pdf`)**: Detailed report for the system architecture and evaluation results.
# Deployment

This repo includes a simple Streamlit app in `app.py`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Docker

```bash
docker build -t traffic-violation-demo .
docker run -p 8501:8501 traffic-violation-demo
```

## Streamlit Cloud

1. Push this repo to GitHub.
2. Go to Streamlit Community Cloud.
3. Select this repository.
4. Set the main file path to `app.py`.
5. Deploy.

Note: the model files are already bundled under `Group28/models/`, so the app does not need to download weights at runtime.
