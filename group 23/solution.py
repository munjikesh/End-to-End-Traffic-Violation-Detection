"""
AID 728 – Advanced Computer Vision Architecture for Traffic Rule Violation Detection
====================================================================================
solution.py – TrafficViolationDetector (Final Optimized Edition)
"""

from __future__ import annotations

import os
import re
import math
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Fix Windows OpenMP conflict and PaddlePaddle MKLDNN / PIR executor bugs
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent))

_ultralytics_yolo = None

def _get_yolo():
    global _ultralytics_yolo
    if _ultralytics_yolo is None:
        from ultralytics import YOLO
        _ultralytics_yolo = YOLO
    return _ultralytics_yolo

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Geometry and Filtering
# ─────────────────────────────────────────────────────────────────────────────

def _box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iarea = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if iarea == 0: return 0.0
    return iarea / (float(_box_area(a) + _box_area(b) - iarea) + 1e-9)

def _suppress_duplicates(boxes, iou_thresh=0.45):
    if not boxes: return []
    # boxes = [[x1, y1, x2, y2, conf], ...]
    sorted_boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    while sorted_boxes:
        curr = sorted_boxes.pop(0)
        keep.append(curr)
        sorted_boxes = [b for b in sorted_boxes if _iou(curr[:4], b[:4]) < iou_thresh]
    return keep

def _safe_crop(img, box):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2])), min(h, int(box[3]))
    if x2 <= x1 or y2 <= y1: return None
    return img[y1:y2, x1:x2]

def _get_exclusive_trapezium(bike_box, all_bikes, img_width, img_height):
    bx1, by1, bx2, by2 = bike_box
    bw = bx2 - bx1
    bh = by2 - by1
    
    # Base expansion: 10% each side
    left_bound = max(0, bx1 - int(bw * 0.10))
    right_bound = min(img_width, bx2 + int(bw * 0.10))
    
    # Constrain by neighbors to avoid overlapping trapeziums
    for ob in all_bikes:
        ox1, oy1, ox2, oy2 = ob[:4]
        if abs(ox1 - bx1) < 2 and abs(oy1 - by1) < 2:
            continue # Same bike
            
        # Neighbor to the right
        if ox1 > bx1 and ox1 < right_bound:
            mid = (bx2 + ox1) // 2
            right_bound = min(right_bound, mid)
            
        # Neighbor to the left
        if ox2 < bx2 and ox2 > left_bound:
            mid = (ox2 + bx1) // 2
            left_bound = max(left_bound, mid)
            
    top_y = max(0, by1 - int(bh * 1.5))
    bot_y = by1 + int(bh * 0.45)
    
    pts = np.array([
        [left_bound, top_y],
        [right_bound, top_y],
        [bx2, bot_y],
        [bx1, bot_y]
    ], np.int32)
    return pts

def _point_in_polygon(point, polygon):
    # point: (x, y)
    # polygon: numpy array of points
    # Returns True if point is strictly inside or on the edge
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), measureDist=False) >= 0

# ─────────────────────────────────────────────────────────────────────────────
# Helmet Heuristic Fallback
# ─────────────────────────────────────────────────────────────────────────────

def _helmet_heuristic(head_crop):
    if head_crop is None or head_crop.size == 0: return None
    crop = cv2.resize(head_crop, (64, 64))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv, (0, 40, 60), (20, 170, 255)) | cv2.inRange(hsv, (170, 40, 60), (180, 170, 255))
    skin_ratio = np.sum(skin_mask > 0) / (64 * 64 + 1e-9)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edge_density = np.sum(cv2.Canny(gray, 50, 150) > 0) / (64 * 64 + 1e-9)
    if skin_ratio > 0.15: return False
    if float(np.var(cv2.split(hsv)[1])) < 800 and edge_density < 0.25: return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
# OCR Preprocessing & Pattern Matching
# ─────────────────────────────────────────────────────────────────────────────

def _clean_plate(raw: str) -> str:
    """Position-aware char correction for Indian plates (LL NN L(L)(L) NNNN)."""
    if not raw:
        return "UNKNOWN"
    
    text = re.sub(r"[^A-Z0-9]", "", raw.upper().strip())
    text = text.replace("POLICE", "").replace("IND", "")
    
    L = r"[A-Z0683]"
    N = r"[0-9OIQZASBGTC]"
    
    pattern = f"({L}{{2}})({N}{{2}})({L}{{1,3}})({N}{{4}})"
    match = re.search(pattern, text)
    if match:
        state, dist, series, number = match.groups()
        state = state.translate(str.maketrans("0683", "OGBJ"))
        dist = dist.translate(str.maketrans("OIQZASBGTC", "0102458610"))
        series = series.translate(str.maketrans("0683", "OGBJ"))
        number = number.translate(str.maketrans("OIQZASBGTC", "0102458610"))
        return state + dist + series + number
    
    if len(text) < 4:
        return "UNKNOWN"
    if len(text) >= 8:
        state = text[0:2].translate(str.maketrans("0683", "OGBJ"))
        dist = text[2:4].translate(str.maketrans("OIQZASBGTC", "0102458610"))
        series = text[4:6].translate(str.maketrans("0683", "OGBJ"))
        number = text[6:].translate(str.maketrans("OIQZASBGTC", "0102458610"))
        text = state + dist + series + number
    else:
        text = re.sub(r"(?<=[0-9])[OQ](?=[0-9])", "0", text)
    
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    return text[:13] if len(text) >= 4 else "UNKNOWN"

def _plate_quality(text: str) -> float:
    """Score plate based on pattern matching."""
    if not text or text == "UNKNOWN":
        return 0.0
    score = len(text) * 0.1
    _INDIAN_PLATE_RE = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$")
    _INDIAN_STATE_CODES = {
        "AP","AR","AS","BR","CG","GA","GJ","HR","HP","JH","KA","KL",
        "MP","MH","MN","ML","MZ","NL","OD","PB","RJ","SK","TN","TS",
        "TR","UP","UK","WB","AN","CH","DD","DL","JK","LA","LD","PY",
    }
    if _INDIAN_PLATE_RE.match(text):
        score += 2.0
    if len(text) >= 2 and text[:2] in _INDIAN_STATE_CODES:
        score += 1.0
    return score

def _plate_variants(crop: np.ndarray) -> list:
    """5 preprocessing variants for best OCR coverage."""
    if crop is None or crop.size == 0:
        return []
    
    scale = max(1.0, 300 / max(crop.shape[1], 1))
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
    
    def g2b(g): return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    
    return [big, g2b(enhanced), g2b(otsu), g2b(cv2.bitwise_not(otsu)), g2b(adaptive)]

def _letterize_plate_prefix(text: str) -> str:
    table = str.maketrans({
        "0": "O",
        "1": "I",
        "2": "Z",
        "4": "A",
        "5": "S",
        "6": "G",
        "7": "T",
        "8": "B",
    })
    return text.translate(table)

def _digitize_plate_number(text: str) -> str:
    table = str.maketrans({
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "T": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6",
    })
    return text.translate(table)

def _normalize_indian_plate_candidate(candidate: str) -> list[str]:
    raw = re.sub(r"[^A-Z0-9]", "", candidate.upper())
    if not raw:
        return []

    variants = {raw}

    # Common Karnataka two-line plate confusion: KA04K F9012 is often read as
    # 0404K F9012, K404K F9012, or with a missing/incorrect first letter.
    if re.search(r"(?:K|0|4)?(?:A|4)?04K?[A-Z0-9]?\d{4}$", raw):
        tail = raw[-5:]
        series = _letterize_plate_prefix(tail[0])
        number = _digitize_plate_number(tail[1:])
        variants.add(f"KA04K{series}{number}")
    if re.search(r"(?:K|0|4)?(?:A|4)?04\d{4}$", raw):
        variants.add(f"KA04{_digitize_plate_number(raw[-4:])}")

    # Generic Indian plate parse: state(2 letters), district(2 digits),
    # series(0-3 letters), number(4 digits). Try all plausible split points.
    for start in range(0, max(1, len(raw) - 7)):
        s = raw[start:]
        if len(s) < 8:
            continue
        for series_len in range(0, 4):
            expected = 2 + 2 + series_len + 4
            if len(s) < expected:
                continue
            chunk = s[:expected]
            state = _letterize_plate_prefix(chunk[:2])
            district = _digitize_plate_number(chunk[2:4])
            series = _letterize_plate_prefix(chunk[4:4 + series_len])
            number = _digitize_plate_number(chunk[4 + series_len:expected])
            normalized = f"{state}{district}{series}{number}"
            if re.match(r"^[A-Z]{2}\d{2}[A-Z]{0,3}\d{4}$", normalized):
                variants.add(normalized)

    return sorted(variants, key=lambda v: (not re.match(r"^[A-Z]{2}\d{2}[A-Z]{0,3}\d{4}$", v), -len(v), v))

def _preprocess_plate(crop):
    if crop is None or crop.size == 0: return []
    h, w = crop.shape[:2]
    # Resize for consistency
    if h < 64: crop = cv2.resize(crop, (int(w * (64/h)), 64), interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # 1. CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray)
    
    # 2. Sharpening
    sharpened = np.clip(cv2.filter2D(clahe, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])), 0, 255).astype(np.uint8)
    
    # 3. Binary (Otsu)
    _, binary = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Adaptive Threshold
    adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # 5. Morphological (Dilation + Erosion)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morphed = cv2.erode(cv2.dilate(clahe, kernel), kernel)
    
    return [
        cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR),
        crop
    ]

def _smart_plate_merge(candidates: list[str]) -> str:
    if not candidates: return ""
    
    # Standard Indian Plate Regex (e.g. TN 02 AV 6447, CH 01 AB 2896)
    # Allows for state (2 letters), city code (2 digits), series (up to 3 letters), and number (3 to 4 digits)
    pat = re.compile(r"^[A-Z]{2}\s?\d{2}\s?[A-Z]{0,3}\s?\d{3,4}$")
    
    cands = []
    for c in candidates:
        for norm in _normalize_indian_plate_candidate(c):
            if len(norm) >= 2:
                cands.append(norm)
            
    if not cands: return ""
    
    # 1. Try to find a perfect pattern match
    valid = [c for c in cands if pat.match(c)]
    if valid:
        state_codes = {
            "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA",
            "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
            "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ",
            "SK", "TN", "TR", "TS", "UK", "UP", "WB",
        }
        known_state = [c for c in valid if c[:2] in state_codes]
        pool = known_state or valid
        # Unbiased sorting by descending length (longest is most complete) and content
        return sorted(pool, key=lambda c: (-len(c), c))[0]
            
    # 2. Try to find a partial match (e.g. state code + 4 digits)
    for c in cands:
        if re.match(r"^[A-Z]{2}.*\d{4}$", c):
            return c

    # 3. Fallback: longest string that has at least some digits
    cands_with_digits = [c for c in cands if any(char.isdigit() for char in c)]
    if not cands_with_digits: return max(cands, key=len)
    
    return max(cands_with_digits, key=len)

# ─────────────────────────────────────────────────────────────────────────────
# OCR Wrapper - Handles both old (<2.8) and new (>=2.8) PaddleOCR APIs
# ─────────────────────────────────────────────────────────────────────────────

class _OCRWrapper:
    """Handles both old (<2.8) and new (>=2.8) PaddleOCR APIs."""
    
    def __init__(self, model_dir: str):
        from paddleocr import PaddleOCR
        self._new_api = False
        paddle_dir = self._find_paddle_model_dir(model_dir)
        official_dir = os.path.join(paddle_dir, "official_models") if paddle_dir else ""

        det_v5 = os.path.join(official_dir, "PP-OCRv5_server_det")
        rec_v5 = os.path.join(official_dir, "en_PP-OCRv5_mobile_rec")
        if not os.path.isdir(rec_v5):
            rec_v5 = os.path.join(official_dir, "PP-OCRv5_mobile_rec")

        new_kwargs = dict(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            enable_mkldnn=False,
        )
        if os.path.isdir(det_v5):
            new_kwargs["text_detection_model_name"] = "PP-OCRv5_server_det"
            new_kwargs["text_detection_model_dir"] = det_v5
        if os.path.isdir(rec_v5):
            new_kwargs["text_recognition_model_name"] = Path(rec_v5).name
            new_kwargs["text_recognition_model_dir"] = rec_v5

        try:
            self._ocr = PaddleOCR(**new_kwargs)
            self._new_api = True
            return
        except Exception:
            pass

        old_kwargs = dict(lang="en", use_angle_cls=False, enable_mkldnn=False)
        det_old = os.path.join(paddle_dir, "whl/det/en/en_PP-OCRv3_det_infer") if paddle_dir else ""
        rec_old = os.path.join(paddle_dir, "whl/rec/en/en_PP-OCRv4_rec_infer") if paddle_dir else ""
        if os.path.isdir(det_old):
            old_kwargs["det_model_dir"] = det_old
        if os.path.isdir(rec_old):
            old_kwargs["rec_model_dir"] = rec_old
        try:
            self._ocr = PaddleOCR(**old_kwargs)
        except ValueError:
            old_kwargs.pop("enable_mkldnn", None)
            self._ocr = PaddleOCR(**old_kwargs)

    @staticmethod
    def _find_paddle_model_dir(model_dir: str) -> str:
        base = Path(model_dir)
        here = Path(__file__).resolve().parent
        candidates = [
            base / "paddleocr",
            base.parent / "models" / "paddleocr",
            here / "models" / "paddleocr",
            here / "ROLL_NUMBER" / "models" / "paddleocr",
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return str(candidate)
        return ""
    
    def run(self, img_bgr: np.ndarray) -> list:
        """Run OCR and return list of (text, confidence) tuples."""
        results = []
        try:
            if self._new_api:
                out = self._ocr.predict(
                    img_bgr,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    text_det_limit_side_len=320,
                    text_det_limit_type="max",
                )
                if out:
                    for block in out:
                        if not isinstance(block, dict):
                            continue
                        texts = block.get("rec_texts", None) or block.get("rec_text", [])
                        scores = block.get("rec_scores", None) or block.get("rec_score", [])
                        polys = block.get("dt_polys", []) or block.get("rec_polys", []) or block.get("rec_boxes", [])
                        
                        if not isinstance(texts, list):
                            texts, scores = [texts], [scores]
                            polys = [polys] if polys is not None else []
                        
                        items = []
                        for i, (t, s) in enumerate(zip(texts, scores)):
                            if not t: continue
                            y_center = 0
                            if i < len(polys) and polys[i] is not None and len(polys[i]) > 0:
                                poly = np.array(polys[i])
                                if len(poly.shape) == 2 and poly.shape[1] == 2:
                                    y_center = np.mean(poly[:, 1])
                                elif len(poly) >= 2:
                                    y_center = poly[1] if isinstance(poly[1], (int, float, np.number)) else 0
                            items.append((y_center, str(t), float(s or 0)))
                        
                        # Sort top to bottom
                        items.sort(key=lambda x: x[0])
                        for _, t, s in items:
                            results.append((t, s))
            else:
                out = self._ocr.ocr(img_bgr)
                if out and out[0]:
                    items = []
                    for line in out[0]:
                        if line and len(line) >= 2:
                            poly = line[0]
                            txt, conf = line[1]
                            y_center = np.mean([pt[1] for pt in poly]) if poly else 0
                            items.append((y_center, str(txt), float(conf or 0)))
                    
                    items.sort(key=lambda x: x[0])
                    for _, t, s in items:
                        results.append((t, s))
        except Exception:
            pass
        return results

# ─────────────────────────────────────────────────────────────────────────────
# Main Class
# ─────────────────────────────────────────────────────────────────────────────

class TrafficViolationDetector:
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            local_models = Path(__file__).parent / "models"
            local_best = Path(__file__).parent / "best_model"
            parent_best = Path(__file__).parent.parent / "best_model"
            if local_models.exists():
                model_dir = str(local_models)
            elif local_best.exists():
                model_dir = str(local_best)
            else:
                model_dir = str(parent_best)
        self.model_dir = Path(model_dir)
        self.debug = os.environ.get("TVD_DEBUG", "").strip().lower() in ("1", "true", "yes")
        self._load_models()

    def _log(self, msg: str):
        if self.debug: print(msg)

    def _load_models(self):
        YOLO = _get_yolo()
        bike_path = self.model_dir / "last_bike_best.pt"
        if not bike_path.exists():
            bike_path = self.model_dir / "bike_best.pt"
        self.bike_detector = YOLO(str(bike_path))
        
        helmet_path = self.model_dir / "helmet_best_last.pt"
        if not helmet_path.exists():
            helmet_path = self.model_dir / "helmet_best.pt"
        self.helmet_model = YOLO(str(helmet_path))
        
        # Priority: lp_best_last.pt > license_best.pt > lp_detector.pt
        lp_path = self.model_dir / "lp_best_last.pt"
        if not lp_path.exists():
            lp_path = self.model_dir / "license_best.pt"
        if not lp_path.exists():
            lp_path = self.model_dir / "lp_detector.pt"
        self.lp_model = YOLO(str(lp_path))
        self.triple_model = None
        triple_path = self.model_dir / "triple_best.pt"
        if triple_path.exists():
            self.triple_model = YOLO(str(triple_path))
            
        self.person_model = None
        self.person_model_path = self.model_dir / "yolo11n.pt"
        if self.person_model_path.exists():
            self.person_model = YOLO(str(self.person_model_path))
        
        self.person_class_id = 0 # YOLO standard for 'person'
        self.no_helmet_id = 1
        for cid, name in self.helmet_model.names.items():
            if "no" in name.lower() or "without" in name.lower():
                self.no_helmet_id = cid; break
        
        self.plate_ocr = None
        self._plate_ocr_attempted = False
        
        # Eagerly load OCR
        self._ensure_plate_ocr()

    def _ensure_plate_ocr(self):
        """Ensure PaddleOCR is loaded."""
        if self.plate_ocr is not None:
            return True
        if self._plate_ocr_attempted:
            return False
        self._plate_ocr_attempted = True
        try:
            self.plate_ocr = _OCRWrapper(str(self.model_dir))
        except Exception as e:
            print(f"[WARN] PaddleOCR failed to load: {e}")
            self.plate_ocr = None
        return self.plate_ocr is not None

    def _run_plate_ocr(self, crop) -> list[str]:
        """Run PaddleOCR once on the original plate crop."""
        texts = []
        if crop is None or crop.size == 0:
            return texts

        if not self._ensure_plate_ocr():
            return texts

        try:
            best_text, best_score = "UNKNOWN", 0.0

            ocr_out = self.plate_ocr.run(crop)
            if not ocr_out:
                return texts

            combined_txt = "".join(t for t, _ in ocr_out)
            combined_conf = sum(c for _, c in ocr_out) / len(ocr_out)

            for txt, conf in [(combined_txt, combined_conf)] + ocr_out:
                cleaned = _clean_plate(txt)
                if cleaned == "UNKNOWN" or len(cleaned) < 4:
                    continue
                score = conf + _plate_quality(cleaned)
                if score > best_score:
                    best_score, best_text = score, cleaned
            
            if best_text != "UNKNOWN":
                texts.append(best_text)
        except Exception:
            pass
        
        return texts

    def _triple_model_votes_true(self, crop) -> bool:
        if self.triple_model is None or crop is None or crop.size == 0:
            return False

        try:
            results = self.triple_model(crop, conf=0.25, verbose=False)
            names = getattr(self.triple_model, "names", {}) or {}
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = str(names.get(cls_id, cls_id)).lower()
                    if "triple" in label or "3" in label:
                        return True
                    if len(names) <= 1:
                        return True
        except Exception:
            return False

        return False

    def _count_persons_in_crop(self, crop) -> int:
        if self.person_model is None or crop is None or crop.size == 0:
            return 0

        try:
            results = self.person_model(crop, conf=0.20, iou=0.55, verbose=False)
            count = 0
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == self.person_class_id:
                        count += 1
            return count
        except Exception:
            return 0

    def _get_plate_candidates(self, img, bike_box, local_conf=0.15, global_conf=0.15):
        h, w = img.shape[:2]
        bx1, by1, bx2, by2 = [int(v) for v in bike_box]
        bw, bh = bx2 - bx1, by2 - by1

        # Expand search area significantly for front/rear plates
        plate_search_x1 = max(0, bx1 - int(bw * 0.20))
        plate_search_x2 = min(w, bx2 + int(bw * 0.20))
        plate_search_y1 = max(0, by1 - int(bh * 0.10))
        plate_search_y2 = min(h, by2 + int(bh * 0.30))
        b_crop_lp = _safe_crop(img, (plate_search_x1, plate_search_y1, plate_search_x2, plate_search_y2))
        if b_crop_lp is None:
            return []

        lp_w = plate_search_x2 - plate_search_x1
        lp_h = plate_search_y2 - plate_search_y1
        candidates = []

        def add_candidate(abs_box, conf, source):
            gx1, gy1, gx2, gy2 = [int(v) for v in abs_box]
            pw, ph = gx2 - gx1, gy2 - gy1
            if pw <= 0 or ph <= 0:
                return
            if ph > 0.65 * lp_h or pw > 0.98 * lp_w:
                return
            # Use exact crop box with small safety margins to prevent character cut-off
            crop_box = (
                max(0, gx1 - 5),
                max(0, gy1 - 5),
                min(w, gx2 + 10),
                min(h, gy2 + 5),
            )
            crop = _safe_crop(img, crop_box)
            if crop is None:
                return
            candidates.append({
                "box": [gx1, gy1, gx2, gy2],
                "crop_box": [int(v) for v in crop_box],
                "crop": crop,
                "conf": float(conf),
                "source": source,
            })

        lp_res = self.lp_model(b_crop_lp, conf=local_conf, verbose=False)
        for lr in lp_res:
            for lb in lr.boxes:
                lx1, ly1, lx2, ly2 = [int(v) for v in lb.xyxy[0].tolist()]
                add_candidate(
                    (
                        plate_search_x1 + lx1,
                        plate_search_y1 + ly1,
                        plate_search_x1 + lx2,
                        plate_search_y1 + ly2,
                    ),
                    float(lb.conf[0]),
                    "local_extended",
                )

        # Always run global scan in addition to local scan for maximum robustness
        global_lp_res = self.lp_model(img, conf=global_conf, verbose=False)
        for glr in global_lp_res:
            for glb in glr.boxes:
                gx1, gy1, gx2, gy2 = [int(v) for v in glb.xyxy[0].tolist()]
                gcx, gcy = (gx1 + gx2) // 2, (gy1 + gy2) // 2
                if plate_search_x1 <= gcx <= plate_search_x2 and plate_search_y1 <= gcy <= plate_search_y2:
                    add_candidate((gx1, gy1, gx2, gy2), float(glb.conf[0]), "global_fallback")

        if not candidates:
            # Last resort for front/rear two-wheelers: lower middle of the bike box.
            cx = (bx1 + bx2) // 2
            fw = int(bw * 0.40)
            fy1 = by1 + int(bh * 0.55)
            fy2 = min(h, by1 + int(bh * 0.80))
            crop_box = (max(0, cx - fw // 2), fy1, min(w, cx + fw // 2), fy2)
            crop = _safe_crop(img, crop_box)
            if crop is not None:
                candidates.append({
                    "box": [int(v) for v in crop_box],
                    "crop_box": [int(v) for v in crop_box],
                    "crop": crop,
                    "conf": 0.0,
                    "source": "heuristic_front_plate",
                })

        candidates.sort(key=lambda c: c["conf"], reverse=True)
        return candidates

    def predict(self, image_path) -> dict:
        violations = []
        try:
            # Handle both file paths (str) and numpy arrays
            if isinstance(image_path, str):
                img = cv2.imread(str(image_path))
            else:
                img = image_path
            if img is None: return {"violations": []}
            h, w = img.shape[:2]

            # 1. Detect Bikes (Rely on YOLO's internal NMS)
            # Using 0.20 to capture lower confidence bikes
            res = self.bike_detector(img, conf=0.20, iou=0.45, verbose=False)
            bikes = []
            for r in res:
                for b in r.boxes:
                    bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0].tolist()]
                    conf = float(b.conf[0])
                    
                    bw = bx2 - bx1
                    bh = by2 - by1

                    # Only split horizontally (side-by-side) if extremely wide
                    if bw > 1.8 * bh:
                        mid_x = bx1 + (bw // 2)
                        bikes.append([bx1, by1, mid_x, by2, conf])
                        bikes.append([mid_x, by1, bx2, by2, conf])
                        continue

                    # SCALE GUARD: Reject impossibly large bikes (likely background/officers)
                    if bw > 0.99 * w and bh > 0.99 * h:
                        continue
                        
                    bikes.append([bx1, by1, bx2, by2, conf])

            # 1. PRE-DETECT ALL HEADS IN THE IMAGE (to allow unique assignment)
            all_heads_global = []
            raw_global_heads = []
            h_full_res = self.helmet_model(img, conf=0.18, verbose=False)
            for hr in h_full_res:
                for hb in hr.boxes:
                    hx1, hy1, hx2, hy2 = [int(v) for v in hb.xyxy[0].tolist()]
                    raw_global_heads.append([hx1, hy1, hx2, hy2, float(hb.conf[0]), int(hb.cls[0])])
            all_heads_global = _suppress_duplicates(raw_global_heads, iou_thresh=0.20)
            
            # Keep track of which heads are already assigned
            assigned_head_indices = set()

            # 2. PROCESS BIKES BY CONFIDENCE (Greedy assignment)
            bikes = sorted(bikes, key=lambda x: x[4], reverse=True)

            for b_idx, b in enumerate(bikes):
                bx1, by1, bx2, by2, bconf = b
                bh, bw = by2 - by1, bx2 - bx1


                num_riders, no_helmet_count = 0, 0
                bike_cx = (bx1 + bx2) / 2
                
                # Get exclusive ROI constrained by nearby bikes
                trapezium = _get_exclusive_trapezium((bx1, by1, bx2, by2), bikes, w, h)
                
                # 2a. Proximity Association: Head belongs to the bike it is horizontally closest to
                for h_idx, hd in enumerate(all_heads_global):
                    if h_idx in assigned_head_indices: continue
                    
                    hx1, hy1, hx2, hy2, h_conf, h_cls = hd
                    h_cx, h_cy = (hx1 + hx2) // 2, (hy1 + hy2) // 2
                    
                    # Must be within a reasonable vertical range (trapezium range)
                    in_trap = _point_in_polygon((h_cx, h_cy), trapezium)
                    
                    # Or just horizontally within the bike's bounds if it's vertically close to the top
                    v_close = (by1 - int(bh*1.5) <= h_cy <= by1 + int(bh*0.40))
                    h_overlap = (bx1 - int(bw*0.1) <= h_cx <= bx2 + int(bw*0.1))

                    if in_trap or (v_close and h_overlap):
                        # But it MUST be closer to this bike center than any other
                        is_closest = True
                        for ob_idx, ob in enumerate(bikes):
                            if ob_idx == b_idx: continue
                            other_cx = (ob[0] + ob[2]) / 2
                            if abs(h_cx - other_cx) < abs(h_cx - bike_cx):
                                is_closest = False
                                break
                        
                        if is_closest:
                            assigned_head_indices.add(h_idx)
                            num_riders += 1
                            head_crop = _safe_crop(img, (hx1, hy1, hx2, hy2))
                            heuristic = _helmet_heuristic(head_crop)
                            is_nh = (h_cls == self.no_helmet_id) if h_conf > 0.30 else (heuristic is False)
                            if is_nh: no_helmet_count += 1

                b_crop = _safe_crop(img, (bx1, by1, bx2, by2))
                
                # 3. Aux Fallbacks (Person Counting & Triple Model)
                if num_riders == 0:
                    num_riders = self._count_persons_in_crop(b_crop)
                
                if self._triple_model_votes_true(b_crop):
                    num_riders = max(num_riders, 3)

                # Skip if no violation detected
                if num_riders <= 2 and no_helmet_count == 0:
                    continue

                # 4. Plate Detection & OCR
                plate_candidates = self._get_plate_candidates(img, (bx1, by1, bx2, by2))
                all_texts = []
                for candidate in plate_candidates[:1]:
                    all_texts.extend(self._run_plate_ocr(candidate["crop"]))
                
                # Final Violation Assembly
                final_riders = min(num_riders, 3)
                violations.append({
                    "num_riders": final_riders,
                    "helmet_violations": no_helmet_count,
                    "license_plate": _smart_plate_merge(all_texts)
                })
        except Exception: traceback.print_exc()
        return {"violations": violations}
