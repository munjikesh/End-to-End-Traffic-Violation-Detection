# Traffic Rule Violation Detection — AID 728 (Group 28)

This directory contains the final submission for **Group 28** for detecting two-wheeler traffic rule violations (helmet violations, triple riding) and recognizing license plates from street images.

We provide **two alternative OCR/violation detection pipelines** packaged cleanly in this submission folder. Neither requires internet access at runtime; all model files are stored locally in the `models/` directory.

---

## 🚀 Two Alternative Pipelines

### 1. PaddleOCR PP-OCRv5 Pipeline (Default: `solution.py`)
This is the default recommended pipeline. It uses custom-trained YOLO weights for vehicle, helmet, and plate detection, combined with an offline PaddleOCR model.
*   **OCR Engine:** PaddleOCR (with offline `PP-OCRv5_server_det` and `en_PP-OCRv5_mobile_rec`).
*   **Optimizations:** Text unwarping, document orientation, and textline orientation modules are disabled to maximize inference speed.
*   **Model loading:** Automatically resolves model paths under `models/` relative to the script location.

### 2. 6-Stage EasyOCR Pipeline (`solution_6stage.py`)
This is the alternative pipeline. It leverages a multi-stage preprocessing pipeline with Test-Time Augmentation (TTA) and EasyOCR.
*   **Preprocessing Stages:** Grayscale conversion ➡️ Contrast normalization (CLAHE) ➡️ High-frequency noise suppression (Bilateral filter) ➡️ Morphological definition (Dilation + Erosion) ➡️ Otsu binarization ➡️ Adaptive thresholding (shadow-resistance) ➡️ Edge sharpening.
*   **Test-Time Augmentation (TTA):** Generates 6 preprocessing variants for each plate crop and aggregates predictions via consensus voting.
*   **OCR Engine:** EasyOCR (using local offline model files).

---

## 🛠️ Directory Structure

```text
Group28/
  solution.py            # PaddleOCR implementation (Default)
  solution_6stage.py     # 6-Stage EasyOCR implementation
  requirements.txt       # Combined package dependencies
  Group28.pdf            # Project report PDF
  README.md              # Subfolder documentation
  models/                # Consolidated local weights (<200 MB total)
    last_bike_best.pt    # Custom two-wheeler detector
    helmet_best_last.pt  # Custom helmet/head detector
    lp_best_last.pt      # Custom license plate detector
    yolo11n.pt           # COCO person detector (fallback)
    paddleocr/           # Local PaddleOCR official weights
      official_models/
        PP-OCRv5_server_det/
        en_PP-OCRv5_mobile_rec/
    easyocr/             # Local EasyOCR model weights
      craft_mlt_25k.pth  # Text detection weights
      english_g2.pth     # Text recognition weights
```

---

## 💻 Python API Usage

Both pipelines expose the exact same `TrafficViolationDetector` interface:

### Running PaddleOCR (Default)
```python
from solution import TrafficViolationDetector

# Initialize (will look for ./models relative to solution.py automatically)
detector = TrafficViolationDetector()

# Stateless prediction on a single image (supports file path or numpy array)
output = detector.predict("path/to/image.jpg")
print(output)
```

### Running 6-Stage EasyOCR
```python
from solution_6stage import TrafficViolationDetector

# Initialize
detector = TrafficViolationDetector()

# Run inference
output = detector.predict("path/to/image.jpg")
print(output)
```

### Output Format
The output is a dictionary listing only violating vehicles:
```json
{
    "violations": [
        {
            "num_riders": 3,
            "helmet_violations": 2,
            "license_plate": "KA04KF9012"
        }
    ]
}
```

---

## 📦 Installation & Dependencies

Install all dependencies required for both versions using pip:

```bash
pip install -r requirements.txt
```
