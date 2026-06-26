from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image


ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "Group28"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from solution import TrafficViolationDetector  # noqa: E402


st.set_page_config(
    page_title="Traffic Violation Detection System",
    page_icon="🛵",
    layout="wide",
)


@st.cache_resource
def load_detector():
    return TrafficViolationDetector()


def _rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[-1] == 3 else image


def _draw_box(img: np.ndarray, box, color, label, thickness=3):
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    out = img.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y0 = max(0, y1 - th - 10)
        cv2.rectangle(out, (x1, y0), (x1 + tw + 10, y1), color, -1)
        cv2.putText(out, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def _run_pipeline(detector, img: np.ndarray):
    stages = {"1. Original": img.copy()}

    persons, motos = detector._detect_objects(img)
    stage2 = img.copy()
    for b in motos:
        stage2 = _draw_box(stage2, b, (0, 180, 255), "bike")
    for p in persons:
        stage2 = _draw_box(stage2, p, (0, 255, 0), "person")
    stages["2. Vehicle detection"] = stage2

    persons = detector._refine_person_detections(img, persons, motos)
    stage3 = img.copy()
    for b in motos:
        stage3 = _draw_box(stage3, b, (0, 180, 255), "bike")
    for p in persons:
        stage3 = _draw_box(stage3, p, (255, 120, 0), "rider?")
        status = detector._classify_helmet(img, p)
        x1, y1, *_ = [int(v) for v in p[:4]]
        cv2.putText(stage3, status, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    stages["3. Rider + helmet"] = stage3

    stage4 = img.copy()
    for i, moto in enumerate(motos):
        stage4 = _draw_box(stage4, moto, (0, 180, 255), f"bike {i+1}")
        plate_text = detector._detect_plate(img, moto)
        if plate_text:
            cv2.putText(
                stage4,
                plate_text,
                (int(moto[0]), max(20, int(moto[1]) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2,
            )
    stages["4. Plate search"] = stage4

    result = detector.predict(img)
    final_annotated = img.copy()
    for i, v in enumerate(result.get("violations", []), start=1):
        if i - 1 < len(motos):
            final_annotated = _draw_box(final_annotated, motos[i - 1], (0, 0, 255), f"violation {i}")
    stages["5. Final output"] = final_annotated
    return stages, result


def main():
    st.title("Traffic Violation Detection System")
    st.caption("PaddleOCR + YOLO based inference with step-by-step visual output for interviews and demos.")

    with st.sidebar:
        st.header("Pipeline")
        st.write("1. Vehicle detection")
        st.write("2. Rider association")
        st.write("3. Helmet classification")
        st.write("4. License plate search")
        st.write("5. Final annotated output")
        st.divider()
        st.write("Models load locally from `Group28/models/`.")

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is None:
        st.info("Upload an image to start.")
        return

    image = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(image, use_container_width=True)

    detector = load_detector()
    with st.spinner("Running detection..."):
        stages, result = _run_pipeline(detector, image_np)

    with col2:
        st.subheader("Result")
        st.json(result)

    violations = result.get("violations", [])
    if violations:
        st.success(f"Found {len(violations)} violating vehicle(s).")
        for idx, v in enumerate(violations, start=1):
            st.write(
                f"{idx}. Riders: {v.get('num_riders', 'N/A')}, "
                f"Helmet violations: {v.get('helmet_violations', 'N/A')}, "
                f"Plate: {v.get('license_plate', 'UNKNOWN')}"
            )
    else:
        st.info("No violations found.")

    final_img = stages["5. Final output"]
    final_pil = Image.fromarray(_rgb(final_img))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        final_pil.save(tmp.name)
        with open(tmp.name, "rb") as f:
            st.download_button(
                "Download annotated image",
                data=f.read(),
                file_name="traffic_violation_annotated.png",
                mime="image/png",
            )

    st.divider()
    st.subheader("Step-by-step pipeline")
    stage_items = list(stages.items())
    progress = st.progress(0)
    for idx, (title, frame) in enumerate(stage_items, start=1):
        st.write(title)
        st.image(_rgb(frame), use_container_width=True)
        progress.progress(int((idx / len(stage_items)) * 100))


if __name__ == "__main__":
    main()
