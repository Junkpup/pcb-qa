"""
core/detector.py — Hybrid PCB defect detector.

YOLODetector   — primary: runs best.pt on the raw live frame.
                 Model classes (confirmed): missing_hole, mouse_bite,
                 open_circuit, short, spur.
                 'spurious_copper' is NOT in the model — handled by diff only.

DiffDetector   — secondary: ORB-aligned diff map. Detects spurious_copper
                 (live brighter than reference = extra copper) and catches
                 any anomaly YOLO misses.

HybridDetector — merges both; YOLO boxes suppress overlapping diff boxes.
                 Exclusion zones (e.g. Camo watermark region) are filtered
                 from both detectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from config import Cfg


# ── Defect data class ─────────────────────────────────────────────────────────

@dataclass
class Defect:
    label:      str
    severity:   str          # "HIGH" | "MEDIUM" | "LOW"
    bbox:       tuple        # (x, y, w, h) in pixels
    area:       float
    confidence: float = 1.0
    source:     str = "diff" # "yolo" | "diff"

    COLORS = {
        "HIGH":   (0,   0,   255),
        "MEDIUM": (0,  165,  255),
        "LOW":    (0,  255,  255),
    }

    @property
    def color(self) -> tuple:
        return self.COLORS.get(self.severity, (255, 255, 255))

    def as_row(self, timestamp: str, frame_n: int) -> list:
        x, y, w, h = self.bbox
        return [timestamp, frame_n, self.label, self.severity,
                x, y, w, h, int(self.area), round(self.confidence, 3), self.source]


# ── Exact severity table for best.pt classes ─────────────────────────────────
# Derived from confirmed model class names:
# {0: 'missing_hole', 1: 'mouse_bite', 2: 'open_circuit', 3: 'short', 4: 'spur'}

_CLASS_SEVERITY: dict[str, str] = {
    "missing_hole":   "HIGH",
    "mouse_bite":     "HIGH",
    "open_circuit":   "HIGH",
    "short":          "HIGH",
    "spur":           "MEDIUM",
    # diff-only class:
    "spurious_copper": "MEDIUM",
}


# ── Exclusion zone helper ─────────────────────────────────────────────────────

def _in_exclusion_zone(bbox: tuple, frame_w: int, frame_h: int,
                        zones: tuple) -> bool:
    """
    Return True if the centre of bbox (x,y,w,h) falls inside any exclusion zone.
    Zones are expressed as (x1_frac, y1_frac, x2_frac, y2_frac) in 0–1 coords.
    """
    x, y, w, h = bbox
    cx = (x + w / 2) / frame_w
    cy = (y + h / 2) / frame_h
    for x1f, y1f, x2f, y2f in zones:
        if x1f <= cx <= x2f and y1f <= cy <= y2f:
            return True
    return False


# ── YOLO detector ─────────────────────────────────────────────────────────────

class YOLODetector:
    """Runs the PCB-trained YOLOv8 model on a frame."""

    def __init__(self, cfg: Cfg):
        self._cfg   = cfg
        self._model = YOLO(str(cfg.MODEL_PATH))
        actual = list(self._model.names.values())
        print(f"[yolo] ✓ Model loaded — classes: {actual}")
        # Warn if model classes differ from what we expect
        expected = set(cfg.YOLO_CLASS_NAMES)
        if set(actual) != expected:
            print(f"[yolo] ⚠  Expected {sorted(expected)}, got {sorted(actual)}")

    def detect(self, frame: np.ndarray) -> list[Defect]:
        results = self._model(
            frame,
            conf=self._cfg.YOLO_CONF,
            iou=self._cfg.YOLO_IOU,
            verbose=False,
        )
        defects: list[Defect] = []
        boxes = results[0].boxes
        if boxes is None:
            return defects

        fh, fw = frame.shape[:2]
        zones  = self._cfg.EXCLUSION_ZONES

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h  = x2 - x1, y2 - y1
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = self._model.names[cls]

            # Reject detections inside exclusion zones (e.g. Camo logo)
            if zones and _in_exclusion_zone((x1, y1, w, h), fw, fh, zones):
                continue

            severity = _CLASS_SEVERITY.get(label, "MEDIUM")
            defects.append(Defect(
                label=label,
                severity=severity,
                bbox=(x1, y1, w, h),
                area=float(w * h),
                confidence=conf,
                source="yolo",
            ))
        return defects


# ── Diff-based detector ───────────────────────────────────────────────────────

class DiffDetector:
    """
    Pixel-difference detector against the ORB-aligned reference frame.

    Defect heuristics (in priority order):
    1. spurious_copper — live region is significantly BRIGHTER than reference
       (extra copper or solder that should not be there).
    2. short           — extreme aspect ratio bright blob (thin bridge).
    3. missing_hole    — live region is significantly DARKER than reference
       (hole blocked or component absent).
    4. open_circuit    — very dark in both ref and live (broken trace/burnt).
    5. mouse_bite      — small irregular blob at board edge (low confidence;
       hard to distinguish from contamination without edge context).
    6. Anomaly         — unclassified change above threshold.
    """

    def __init__(self, cfg: Cfg):
        self._cfg = cfg

    def detect(
        self,
        reference: np.ndarray,
        aligned: np.ndarray,
    ) -> tuple[list[Defect], np.ndarray]:
        """Returns (defects, diff_map_bgr)."""
        blur = self._cfg.DIFF_BLUR
        ref_g  = cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY),
                                   (blur, blur), 0)
        live_g = cv2.GaussianBlur(cv2.cvtColor(aligned,   cv2.COLOR_BGR2GRAY),
                                   (blur, blur), 0)

        # Compensate for global exposure drift (auto-exposure between ref capture
        # and live feed). Without this, a dimmer live frame makes every pixel
        # appear "dark" and the classifier floods everything as missing_hole.
        ref_mean  = float(ref_g.mean())
        live_mean = float(live_g.mean())
        if live_mean > 1.0:
            scale  = ref_mean / live_mean
            live_g = np.clip(live_g.astype(np.float32) * scale, 0, 255).astype(np.uint8)

        diff = cv2.absdiff(ref_g, live_g)
        _, thresh = cv2.threshold(diff, self._cfg.DIFF_THRESH, 255, cv2.THRESH_BINARY)

        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (self._cfg.MORPH_OPEN, self._cfg.MORPH_OPEN))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (self._cfg.MORPH_CLOSE, self._cfg.MORPH_CLOSE))
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k_open)
        clean = cv2.morphologyEx(clean,  cv2.MORPH_CLOSE, k_close)

        diff_map = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

        fh, fw  = reference.shape[:2]
        zones   = self._cfg.EXCLUSION_ZONES
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        defects: list[Defect] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._cfg.MIN_DEFECT_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Skip blobs inside exclusion zones (Camo watermark, etc.)
            if zones and _in_exclusion_zone((x, y, w, h), fw, fh, zones):
                continue

            aspect    = w / max(h, 1)
            mean_ref  = float(np.mean(ref_g[y:y+h, x:x+w]))
            mean_live = float(np.mean(live_g[y:y+h, x:x+w]))
            delta     = mean_live - mean_ref   # positive = live brighter

            # ── Classification ────────────────────────────────────────────
            if delta > 25 and not (aspect > 3.5 or aspect < 0.28):
                # Live is brighter and not thin = extra copper on board
                label    = "spurious_copper"
                severity = "MEDIUM"
                conf     = min(1.0, delta / 60)

            elif aspect > 3.5 or aspect < 0.28:
                # Thin elongated blob = solder bridge / short (diff can see it too)
                label    = "short"
                severity = "HIGH"
                conf     = min(1.0, area / 1500)

            elif delta < -20:
                # Live darker than reference = hole / missing component
                label    = "missing_hole"
                severity = "HIGH"
                conf     = min(1.0, abs(delta) / 80)

            elif mean_live < 35 and mean_ref < 35:
                # Both very dark = open circuit / burnt trace
                label    = "open_circuit"
                severity = "HIGH"
                conf     = 0.75

            else:
                # Unclassified change
                label    = "anomaly"
                severity = "LOW"
                conf     = 0.50

            if conf < self._cfg.DIFF_MIN_CONF:
                continue

            defects.append(Defect(label, severity, (x, y, w, h), area, conf, source="diff"))

        return defects, diff_map


# ── Hybrid merger ─────────────────────────────────────────────────────────────

class HybridDetector:
    """
    Runs YOLO and diff detectors then merges results.
    YOLO boxes are authoritative; diff boxes that overlap a YOLO box
    by IoU ≥ MERGE_IOU_THRESH are suppressed to avoid duplicates.
    """

    def __init__(self, cfg: Cfg):
        self._cfg  = cfg
        self._yolo = YOLODetector(cfg)
        self._diff = DiffDetector(cfg)

    def detect(
        self,
        live_frame: np.ndarray,
        reference: Optional[np.ndarray],
        aligned: Optional[np.ndarray],
    ) -> tuple[list[Defect], Optional[np.ndarray]]:
        """
        Returns (merged_defects, diff_map_bgr).
        diff_map is None when no aligned frame is available.
        """
        yolo_defects = self._yolo.detect(live_frame)

        diff_map: Optional[np.ndarray] = None
        diff_defects: list[Defect] = []

        if reference is not None and aligned is not None:
            diff_defects, diff_map = self._diff.detect(reference, aligned)

        merged = list(yolo_defects)
        thresh = self._cfg.MERGE_IOU_THRESH

        for d in diff_defects:
            if not any(_iou(d.bbox, y.bbox) >= thresh for y in yolo_defects):
                merged.append(d)

        return merged, diff_map


# ── IoU helper ────────────────────────────────────────────────────────────────

def _iou(a: tuple, b: tuple) -> float:
    """Intersection over Union for two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
