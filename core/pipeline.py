"""
core/pipeline.py — InspectionPipeline QThread.

Owns the camera, reference manager, aligner, detector, and logger.
Emits Qt signals consumed by the UI; never blocks the main thread.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from config import Cfg
from core.camera import CameraSource
from core.reference import ReferenceManager
from core.aligner import FeatureMatcher, HomographyAligner
from core.detector import HybridDetector, Defect
from core.logger import CSVLogger


def _box_iou(a: tuple, b: tuple) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax1+aw, bx1+bw); iy2 = min(ay1+ah, by1+bh)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    return inter / (aw*ah + bw*bh - inter)


class _DefectTracker:
    """
    IoU-based temporal smoother.
    A detection must persist at the same location for `required` consecutive
    frames before it is treated as a real defect.
    """

    def __init__(self, iou_thresh: float = 0.25, required: int = 3):
        self._iou_thresh = iou_thresh
        self._required   = required
        # list of [defect, consecutive_hit_count]
        self._tracked: list[list] = []

    def reset(self) -> None:
        self._tracked = []

    def update(self, detections: list[Defect]) -> list[Defect]:
        new_tracked: list[list] = []
        used = set()

        for entry in self._tracked:
            prev_d, count = entry
            best_iou, best_i = 0.0, -1
            for i, d in enumerate(detections):
                if i in used or d.label != prev_d.label:
                    continue
                iou = _box_iou(d.bbox, prev_d.bbox)
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_iou >= self._iou_thresh:
                used.add(best_i)
                new_tracked.append([detections[best_i], count + 1])
            # else: defect disappeared — drop it

        # New candidates (not matched to any existing track)
        for i, d in enumerate(detections):
            if i not in used:
                new_tracked.append([d, 1])

        self._tracked = new_tracked
        return [entry[0] for entry in new_tracked if entry[1] >= self._required]


class InspectionPipeline(QThread):
    """
    Runs the full inspection loop in a background thread.

    Signals
    -------
    frame_ready(np.ndarray)   — latest annotated BGR frame for display
    defects_ready(list)       — list[Defect] from the last detection pass
    status_update(dict)       — telemetry: fps, match_count, reproj_err, verdict
    error(str)                — human-readable error message
    """

    frame_ready:   pyqtSignal = pyqtSignal(object)
    defects_ready: pyqtSignal = pyqtSignal(object)
    status_update: pyqtSignal = pyqtSignal(dict)
    error:         pyqtSignal = pyqtSignal(str)

    def __init__(self, cfg: Cfg, camera: CameraSource, parent=None):
        super().__init__(parent)
        self._cfg      = cfg
        self._camera   = camera
        self._ref      = ReferenceManager(cfg)
        self._matcher  = FeatureMatcher(cfg)
        self._aligner  = HomographyAligner(cfg)
        self._detector = HybridDetector(cfg)
        self._logger   = CSVLogger(cfg)

        self._running       = False
        self._inspecting    = False
        self._show_grid     = True
        self._show_diff     = False
        self._frame_n       = 0
        self._display_tick  = 0
        self._pending_ref   = False   # safe to call capture_reference() before run()

        self._tracker = _DefectTracker(
            iou_thresh=0.25,
            required=cfg.TEMPORAL_FRAMES,
        )

        # Latest state (read from UI thread via properties)
        self._last_defects: list[Defect] = []
        self._diff_map: Optional[np.ndarray] = None

    # ── Public control API (called from UI thread) ────────────────────────────

    def stop(self) -> None:
        self._running = False

    def capture_reference(self) -> None:
        """Request reference capture on the next frame."""
        self._pending_ref = True

    def set_inspecting(self, value: bool) -> None:
        self._inspecting = value
        if not value:
            self._last_defects = []
            self._tracker.reset()

    def set_show_grid(self, value: bool) -> None:
        self._show_grid = value

    def set_show_diff(self, value: bool) -> None:
        self._show_diff = value

    @property
    def reference(self) -> ReferenceManager:
        return self._ref

    @property
    def last_log_path(self) -> str:
        return str(self._logger.path)

    # ── QThread entry point ───────────────────────────────────────────────────

    def run(self) -> None:
        self._running    = True
        self._pending_ref = False

        fps_counter = 0
        fps_val     = 0.0
        fps_t       = time.time()
        match_count = 0
        reproj_err  = 0.0
        verdict     = ""

        print("[pipeline] ✓ Inspection loop started.")

        while self._running:
            frame = self._camera.read()
            if frame is None:
                self.error.emit("Camera read failed — check connection.")
                time.sleep(0.05)
                continue

            self._frame_n += 1
            fps_counter   += 1
            if fps_counter >= 30:
                fps_val     = fps_counter / (time.time() - fps_t + 1e-9)
                fps_t       = time.time()
                fps_counter = 0

            display = frame.copy()

            # ── Reference capture ─────────────────────────────────────────
            if self._pending_ref:
                self._ref.set(frame)
                self._pending_ref = False
                self._last_defects = []
                self._tracker.reset()
                verdict = ""
                match_count = 0
                reproj_err  = 0.0

            # ── Inspection pipeline ───────────────────────────────────────
            if self._inspecting:
                warped: Optional[np.ndarray] = None

                # Alignment (only possible when a reference has been captured)
                if self._ref.ready:
                    live_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    src_pts, dst_pts, match_count = self._matcher.match(self._ref, live_gray)

                    if src_pts is not None and match_count >= 4:
                        warped, reproj_err = self._aligner.align(
                            frame, src_pts, dst_pts, self._ref.frame.shape)
                    else:
                        # Alignment failed — warn but still run YOLO below
                        verdict = "ALIGN_FAIL"
                        cv2.putText(
                            display,
                            f"Low matches ({match_count}) — move PCB closer / improve lighting",
                            (10, self._cfg.OUTPUT_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2,
                        )

                # YOLO always runs; diff only runs when warped is available
                ref_frame = self._ref.frame if self._ref.ready else None
                raw_defects, diff_map = self._detector.detect(frame, ref_frame, warped)

                # Temporal filtering: only apply when a reference is available.
                # YOLO-only mode (no reference) shows detections immediately.
                # With a reference, diff detections are temporally smoothed to
                # reduce false positives from alignment jitter and noise.
                if self._ref.ready:
                    confirmed = self._tracker.update(raw_defects)
                else:
                    # No reference — only YOLO is running. Show YOLO results immediately.
                    confirmed = raw_defects

                # The video overlay shows raw detections (immediate visual feedback);
                # only confirmed defects are logged and pushed to the UI panel.
                self._last_defects = raw_defects
                self._diff_map     = diff_map
                if verdict != "ALIGN_FAIL":
                    verdict = "FAIL" if confirmed else "PASS"

                if confirmed:
                    self._logger.log(confirmed, self._frame_n)

                self.defects_ready.emit(confirmed)

            # ── Draw defect annotations ───────────────────────────────────
            for d in self._last_defects:
                x, y, w, h = d.bbox
                cv2.rectangle(display, (x, y), (x+w, y+h), d.color, 2)
                label = f"{d.label}  {d.severity}  {d.confidence:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                cv2.rectangle(display, (x, y-th-6), (x+tw+4, y), (20, 20, 20), -1)
                cv2.putText(display, label, (x+2, y-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, d.color, 1, cv2.LINE_AA)

            # ── Grid overlay ──────────────────────────────────────────────
            if self._show_grid:
                display = self._draw_grid(display)

            # ── Diff map inset ────────────────────────────────────────────
            if self._show_diff and self._diff_map is not None:
                dh = display.shape[0] // 3
                dw = display.shape[1] // 3
                inset = cv2.resize(self._diff_map, (dw, dh))
                display[0:dh, 0:dw] = inset

            # ── HUD ───────────────────────────────────────────────────────
            display = self._draw_hud(display, {
                "mode":        "INSPECT" if self._inspecting else "VIEW",
                "ref_ready":   self._ref.ready,
                "fps":         fps_val,
                "match_count": match_count,
                "reproj_err":  reproj_err,
                "n_defects":   len(self._last_defects),
                "verdict":     verdict if self._inspecting else "",
            })

            # Only push a frame to Qt every DISPLAY_INTERVAL frames —
            # detection still runs every frame; this just cuts rendering cost.
            self._display_tick += 1
            if self._display_tick >= self._cfg.DISPLAY_INTERVAL:
                self._display_tick = 0
                self.frame_ready.emit(display)

            self.status_update.emit({
                "fps":         fps_val,
                "match_count": match_count,
                "reproj_err":  reproj_err,
                "verdict":     verdict,
                "n_defects":   len(self._last_defects),
                "ref_ready":   self._ref.ready,
            })

        print(f"[pipeline] Session ended — {self._frame_n} frames processed.")

    # ── Private drawing helpers ───────────────────────────────────────────────

    def _draw_grid(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        h, w = frame.shape[:2]
        for x in range(0, w, self._cfg.GRID_PX):
            cv2.line(overlay, (x, 0), (x, h), self._cfg.GRID_COLOR, 1)
        for y in range(0, h, self._cfg.GRID_PX):
            cv2.line(overlay, (0, y), (w, y), self._cfg.GRID_COLOR, 1)
        return cv2.addWeighted(overlay, self._cfg.GRID_ALPHA,
                               frame,  1 - self._cfg.GRID_ALPHA, 0)

    def _draw_hud(self, frame: np.ndarray, state: dict) -> np.ndarray:
        out  = frame.copy()
        h, w = frame.shape[:2]
        bar_h = 36
        cv2.rectangle(out, (0, h - bar_h), (w, h), (15, 15, 15), -1)

        bottom = h - 10
        pieces = [
            (f"[{state['mode']}]",     8,   (200, 200, 200)),
            ("REF ✓" if state["ref_ready"] else "NO REF",
                                       110, (0, 255, 0) if state["ref_ready"] else (80, 80, 200)),
            (f"FPS {state['fps']:.1f}", 210, (200, 200, 200)),
        ]
        if state["ref_ready"]:
            pieces += [
                (f"Matches {state['match_count']}",    310, (200, 200, 200)),
                (f"Reproj {state['reproj_err']:.1f}px", 460, (200, 200, 200)),
            ]

        for text, x, color in pieces:
            cv2.putText(out, text, (x, bottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        verdict = state.get("verdict", "")
        if verdict:
            nd = state.get("n_defects", 0)
            v_color = (0, 220, 0) if verdict == "PASS" else \
                      (0, 0, 220) if verdict == "FAIL" else (180, 180, 180)
            cv2.putText(out,
                        f"{verdict}  ({nd} defect{'s' if nd != 1 else ''})",
                        (w - 260, bottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, v_color, 2)

        cv2.putText(out,
                    "R=ref  I=inspect  G=grid  D=diff  S=save  Q=quit",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)
        return out
