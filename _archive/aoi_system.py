"""
AOI System — Complete Rewrite
==============================
macOS  |  scrcpy ≥ 2.1  |  Python 3.9+

HOW IT WORKS
------------
scrcpy streams your phone camera as raw H.264 to stdout.
ffmpeg decodes that stream into raw BGR frames, also to stdout.
This script reads those bytes directly — no screen capture, no mss,
no window detection. OpenCV gets clean BGR frames identical to what
cv2.VideoCapture would give you.

PIPELINE STAGES
---------------
  STAGE 0  Live feed — just show the camera, verify it works.
  STAGE 1  Press R to freeze a reference frame (known-good PCB).
  STAGE 2  Every subsequent frame is matched to the reference via ORB.
  STAGE 3  Homography warps the live frame onto the reference plane.
  STAGE 4  Absolute difference → threshold → contour analysis → defects.
  STAGE 5  Defects logged to CSV, annotated on screen.

SETUP (run once)
----------------
  brew install scrcpy ffmpeg
  pip install opencv-python numpy

  On your phone:
    Settings → Developer Options → USB Debugging ON
    Plug in via USB and accept the debug prompt.
    Confirm with:  adb devices   (should show "device", not "unauthorized")

RUN
---
  python aoi_system.py

  Optional flags:
    --ref path/to/reference.png   skip live reference capture, load file
    --width 1280                  output frame width  (default 1280)
    --height 720                  output frame height (default 720)
    --camera-id 0                 phone camera index  (default 0)
    --no-inspect                  start in view-only mode

CONTROLS (keyboard, OpenCV window must be in focus)
----------------------------------------------------
  R          Capture current frame as reference (starts inspection)
  I          Toggle inspection mode on / off
  G          Toggle grid overlay
  D          Toggle diff-map side panel
  S          Save annotated snapshot to ./snapshots/
  Q / ESC    Quit
"""

import subprocess
import threading
import queue
import time
import sys
import signal
import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  (all tuneable values in one place)
# ──────────────────────────────────────────────────────────────────────────────

class Cfg:
    # ── scrcpy / stream ───────────────────────────────────────────────────────
    CAMERA_ID        = 0
    CAMERA_RES       = "1920x1080"    # source capture resolution
    BITRATE          = "8M"
    OUTPUT_W         = 1280
    OUTPUT_H         = 720

    # ── grid overlay ─────────────────────────────────────────────────────────
    GRID_PX          = 100            # grid cell size in pixels
    GRID_COLOR       = (0, 255, 0)    # BGR green
    GRID_ALPHA       = 0.20

    # ── ORB feature matching ──────────────────────────────────────────────────
    ORB_FEATURES     = 5000
    ORB_MIN_MATCHES  = 30             # warn if fewer good matches
    LOWE_RATIO       = 0.75

    # ── homography validation ─────────────────────────────────────────────────
    H_DET_MIN        = 0.05
    H_DET_MAX        = 20.0
    H_REPROJ_MAX     = 4.0            # px RMS reprojection error ceiling

    # ── defect detection ──────────────────────────────────────────────────────
    DIFF_BLUR        = 5              # GaussianBlur ksize before diff
    DIFF_THRESH      = 40             # absolute pixel diff threshold
    MORPH_OPEN       = 3              # morphological opening kernel size
    MORPH_CLOSE      = 7             # morphological closing kernel size
    MIN_DEFECT_AREA  = 200            # px² — blobs smaller than this = noise

    # ── I/O ───────────────────────────────────────────────────────────────────
    SNAPSHOT_DIR     = Path("snapshots")
    CSV_PATH         = Path("aoi_defect_log.csv")
    CSV_FIELDS       = ["timestamp", "frame", "label", "severity",
                        "x", "y", "w", "h", "area", "confidence"]


# ──────────────────────────────────────────────────────────────────────────────
# DEFECT DATA CLASS
# ──────────────────────────────────────────────────────────────────────────────

class Defect:
    __slots__ = ("label", "severity", "bbox", "area", "confidence")

    COLORS = {
        "HIGH":   (0,   0,   255),
        "MEDIUM": (0,  165,  255),
        "LOW":    (0,  255,  255),
    }

    def __init__(self, label, severity, bbox, area, confidence=1.0):
        self.label      = label
        self.severity   = severity   # "HIGH" | "MEDIUM" | "LOW"
        self.bbox       = bbox       # (x, y, w, h)
        self.area       = area
        self.confidence = confidence

    @property
    def color(self):
        return self.COLORS.get(self.severity, (255, 255, 255))

    def as_row(self, ts, frame_n):
        x, y, w, h = self.bbox
        return [ts, frame_n, self.label, self.severity,
                x, y, w, h, int(self.area), round(self.confidence, 3)]


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 0 — SCRCPY → FFMPEG → OPENCV PIPE
# ──────────────────────────────────────────────────────────────────────────────

class VideoStream:
    """
    Launches scrcpy and ffmpeg as subprocesses, reads raw BGR frames
    from ffmpeg stdout in a background thread, and exposes them via
    read() — exactly like cv2.VideoCapture but from your phone camera.
    """

    def __init__(self, cfg: Cfg):
        self.cfg         = cfg
        self._q          = queue.Queue(maxsize=2)
        self._scrcpy     : Optional[subprocess.Popen] = None
        self._ffmpeg     : Optional[subprocess.Popen] = None
        self._thread     : Optional[threading.Thread] = None
        self._alive      = threading.Event()
        self._frame_size = cfg.OUTPUT_W * cfg.OUTPUT_H * 3   # BGR bytes

    # ── public ────────────────────────────────────────────────────────────────

    def start(self):
        self._alive.set()
        self._scrcpy = self._launch_scrcpy()

        print("[stream]  Waiting 3 s for scrcpy to negotiate with device...")
        time.sleep(3)

        if self._scrcpy.poll() is not None:
            self._diagnose_scrcpy_fail()
            sys.exit(1)

        self._ffmpeg = self._launch_ffmpeg()
        time.sleep(0.5)

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        print("[stream]  Frame reader thread started.")

    def read(self, timeout=2.0):
        """Return latest BGR frame or None on timeout."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._alive.clear()
        for proc, name in [(self._ffmpeg, "ffmpeg"), (self._scrcpy, "scrcpy")]:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=4)
                    print(f"[stream]  {name} terminated.")
                except subprocess.TimeoutExpired:
                    proc.kill()
                    print(f"[stream]  {name} killed (timeout).")

    # ── internal ──────────────────────────────────────────────────────────────

    def _launch_scrcpy(self):
        cmd = [
            "scrcpy",
            "--video-source=camera",
            f"--camera-id={self.cfg.CAMERA_ID}",
            f"--camera-size={self.cfg.CAMERA_RES}",
            f"--video-bit-rate={self.cfg.BITRATE}",
            "--no-display",        # ← DO NOT open any scrcpy window
            "--video-codec=h264",
            "-",                   # write encoded video to stdout
        ]
        print(f"[scrcpy]  {' '.join(cmd)}")
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def _launch_ffmpeg(self):
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-f",        "h264",        # input format
            "-i",        "pipe:0",      # read from stdin
            "-f",        "rawvideo",    # output format
            "-pix_fmt",  "bgr24",       # OpenCV native pixel format
            "-vf",       f"scale={self.cfg.OUTPUT_W}:{self.cfg.OUTPUT_H}",
            "-",                        # write raw frames to stdout
        ]
        print(f"[ffmpeg]  {' '.join(cmd)}")
        return subprocess.Popen(
            cmd,
            stdin=self._scrcpy.stdout,  # pipe scrcpy stdout → ffmpeg stdin
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def _reader(self):
        """Background thread: read raw BGR frames from ffmpeg stdout."""
        while self._alive.is_set():
            try:
                raw = self._ffmpeg.stdout.read(self._frame_size)
            except Exception as e:
                print(f"[reader]  Read error: {e}")
                break

            if len(raw) != self._frame_size:
                if self._alive.is_set():
                    print("[reader]  stdout closed — pipeline may have exited.")
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.cfg.OUTPUT_H, self.cfg.OUTPUT_W, 3)
            ).copy()

            # Drop stale frames — always serve the freshest one
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            self._q.put(frame)

    @staticmethod
    def _diagnose_scrcpy_fail():
        print("\n[ERROR]  scrcpy exited immediately. Common causes on macOS:")
        print("  1. Phone not connected or USB Debugging not enabled.")
        print("     → Run:  adb devices   (must show 'device', not 'unauthorized')")
        print("  2. scrcpy version too old (need ≥ 2.1 for --video-source=camera).")
        print("     → Run:  scrcpy --version")
        print("     → Fix:  brew upgrade scrcpy")
        print("  3. Camera permission denied on device.")
        print("     → Disconnect, reconnect, accept the debug prompt on screen.\n")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — REFERENCE MANAGER
# ──────────────────────────────────────────────────────────────────────────────

class ReferenceManager:
    """
    Holds the known-good PCB reference frame and its ORB descriptors.
    Call .set(frame) to capture, .load(path) to load from disk.
    """

    def __init__(self, cfg: Cfg):
        self._orb  = cv2.ORB_create(nfeatures=cfg.ORB_FEATURES)
        self.cfg   = cfg
        self.reset()

    def reset(self):
        self.frame : Optional[np.ndarray] = None
        self.gray  : Optional[np.ndarray] = None
        self.kp    : list = []
        self.desc  : Optional[np.ndarray] = None

    @property
    def ready(self):
        return self.frame is not None and self.desc is not None

    def set(self, frame: np.ndarray):
        self.frame = frame.copy()
        self.gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.kp, self.desc = self._orb.detectAndCompute(self.gray, None)
        n = len(self.kp)
        print(f"[reference] Captured — {n} ORB keypoints.")
        if n < self.cfg.ORB_MIN_MATCHES:
            print(f"[reference] ⚠  Only {n} keypoints. "
                  "Improve lighting or PCB texture for better matching.")
        return self

    def load(self, path: str):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        # Resize to match pipeline output size
        img = cv2.resize(img, (self.cfg.OUTPUT_W, self.cfg.OUTPUT_H))
        self.set(img)
        print(f"[reference] Loaded from {path}")
        return self


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — ORB FEATURE MATCHER
# ──────────────────────────────────────────────────────────────────────────────

class FeatureMatcher:
    """
    Matches ORB keypoints between the reference and a live frame.
    Returns aligned source/destination point arrays for homography.
    """

    def __init__(self, cfg: Cfg):
        self.cfg  = cfg
        self._orb = cv2.ORB_create(nfeatures=cfg.ORB_FEATURES)
        self._bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(self, ref: ReferenceManager, live_gray: np.ndarray):
        """
        Returns (src_pts, dst_pts, n_matches).
        src_pts — points in reference frame
        dst_pts — corresponding points in live frame
        Returns (None, None, 0) if matching fails.
        """
        kp2, desc2 = self._orb.detectAndCompute(live_gray, None)

        if desc2 is None or ref.desc is None or len(kp2) < 2:
            return None, None, 0

        # knnMatch then Lowe's ratio test
        try:
            raw = self._bf.knnMatch(ref.desc, desc2, k=2)
        except cv2.error:
            return None, None, 0

        good = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.cfg.LOWE_RATIO * n.distance:
                    good.append(m)

        if len(good) < 4:          # homography needs ≥ 4 points
            return None, None, len(good)

        src = np.float32([ref.kp[m.queryIdx].pt for m in good])
        dst = np.float32([kp2[m.trainIdx].pt    for m in good])
        return src, dst, len(good)


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3 — HOMOGRAPHY ALIGNER
# ──────────────────────────────────────────────────────────────────────────────

class HomographyAligner:
    """
    Computes a homography H that maps the live frame onto the reference plane,
    warps the live frame, and validates the result.
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def align(self, live_frame: np.ndarray,
              src_pts: np.ndarray, dst_pts: np.ndarray,
              ref_shape: tuple):
        """
        Returns (warped_frame, reproj_error) or (None, inf) on failure.
        """
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        if H is None:
            return None, float("inf")

        # Validate determinant (catches degenerate / flipped homographies)
        det = abs(np.linalg.det(H))
        if not (self.cfg.H_DET_MIN < det < self.cfg.H_DET_MAX):
            print(f"[align]  Rejected H — det={det:.4f}")
            return None, float("inf")

        # Reprojection error on RANSAC inliers
        inliers = mask.ravel() == 1
        if inliers.sum() < 4:
            return None, float("inf")

        src_in  = src_pts[inliers]
        dst_in  = dst_pts[inliers]
        proj    = cv2.perspectiveTransform(
            dst_in.reshape(-1, 1, 2), H).reshape(-1, 2)
        err     = float(np.sqrt(np.mean(np.sum((src_in - proj) ** 2, axis=1))))

        if err > self.cfg.H_REPROJ_MAX:
            print(f"[align]  High reprojection error: {err:.2f} px — skipping frame.")
            return None, err

        h, w = ref_shape[:2]
        warped = cv2.warpPerspective(live_frame, H, (w, h))
        return warped, err


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 4 — DEFECT DETECTOR
# ──────────────────────────────────────────────────────────────────────────────

class DefectDetector:
    """
    Compares an aligned live frame to the reference frame.
    Returns a list of Defect objects and the diff map.
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def detect(self, reference: np.ndarray,
               aligned: np.ndarray) -> tuple:
        """
        Returns (defects: list[Defect], diff_map: np.ndarray).
        """
        ref_g  = cv2.GaussianBlur(
            cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY),
            (self.cfg.DIFF_BLUR, self.cfg.DIFF_BLUR), 0)
        live_g = cv2.GaussianBlur(
            cv2.cvtColor(aligned,   cv2.COLOR_BGR2GRAY),
            (self.cfg.DIFF_BLUR, self.cfg.DIFF_BLUR), 0)

        diff = cv2.absdiff(ref_g, live_g)

        # Threshold
        _, thresh = cv2.threshold(
            diff, self.cfg.DIFF_THRESH, 255, cv2.THRESH_BINARY)

        # Morphological clean-up
        k_open  = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.cfg.MORPH_OPEN, self.cfg.MORPH_OPEN))
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.cfg.MORPH_CLOSE, self.cfg.MORPH_CLOSE))
        clean   = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k_open)
        clean   = cv2.morphologyEx(clean,  cv2.MORPH_CLOSE, k_close)

        # Colourised diff map for display
        diff_map = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

        # Contour analysis → classify defects
        contours, _ = cv2.findContours(
            clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        defects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.cfg.MIN_DEFECT_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect      = w / max(h, 1)
            roi_ref     = ref_g[y:y+h, x:x+w]
            roi_live    = live_g[y:y+h, x:x+w]
            mean_ref    = float(np.mean(roi_ref))
            mean_live   = float(np.mean(roi_live))

            # ── Classification heuristics ─────────────────────────────────
            #
            # Solder bridge:  thin bright blob  (aspect ratio extremes,
            #                 live region is brighter than reference)
            # Missing comp.:  live region is darker than reference
            #                 (component removed → less bright metal)
            # Burnt region:   very dark in both ref and live diff area
            # Contamination:  small blob, moderate brightness difference

            if aspect > 3.0 or aspect < 0.33:
                label    = "Solder Bridge"
                severity = "HIGH"
                conf     = min(1.0, area / 1500)

            elif mean_live < mean_ref - 20:
                label    = "Missing Component"
                severity = "HIGH"
                conf     = min(1.0, (mean_ref - mean_live) / 80)

            elif mean_live < 40 and mean_ref < 40:
                label    = "Burnt Region"
                severity = "HIGH"
                conf     = 0.85

            elif area < 800:
                label    = "Contamination"
                severity = "LOW"
                conf     = 0.6

            else:
                label    = "Anomaly"
                severity = "MEDIUM"
                conf     = 0.7

            defects.append(Defect(label, severity, (x, y, w, h), area, conf))

        return defects, diff_map


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 5 — CSV LOGGER
# ──────────────────────────────────────────────────────────────────────────────

class CSVLogger:
    def __init__(self, cfg: Cfg):
        self.path = cfg.CSV_PATH
        self._write_header()

    def _write_header(self):
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(Cfg.CSV_FIELDS)

    def log(self, defects: list, frame_n: int):
        if not defects:
            return
        ts = datetime.now().isoformat(timespec="seconds")
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            for d in defects:
                w.writerow(d.as_row(ts, frame_n))


# ──────────────────────────────────────────────────────────────────────────────
# RENDERER — draws everything onto the display frame
# ──────────────────────────────────────────────────────────────────────────────

class Renderer:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def draw_grid(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        h, w    = frame.shape[:2]
        for x in range(0, w, self.cfg.GRID_PX):
            cv2.line(overlay, (x, 0), (x, h), self.cfg.GRID_COLOR, 1)
        for y in range(0, h, self.cfg.GRID_PX):
            cv2.line(overlay, (0, y), (w, y), self.cfg.GRID_COLOR, 1)
        return cv2.addWeighted(
            overlay, self.cfg.GRID_ALPHA, frame, 1 - self.cfg.GRID_ALPHA, 0)

    def draw_defects(self, frame: np.ndarray,
                     defects: list) -> np.ndarray:
        out = frame.copy()
        for d in defects:
            x, y, w, h = d.bbox
            cv2.rectangle(out, (x, y), (x+w, y+h), d.color, 2)
            label = f"{d.label}  {d.severity}  {d.confidence:.0%}"
            # Background chip for readability
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(out, (x, y-th-6), (x+tw+4, y), (20, 20, 20), -1)
            cv2.putText(out, label, (x+2, y-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, d.color, 1,
                        cv2.LINE_AA)
        return out

    def draw_hud(self, frame: np.ndarray, state: dict) -> np.ndarray:
        """
        state keys: mode, ref_ready, fps, frame_n,
                    match_count, reproj_err, n_defects, verdict
        """
        out = frame.copy()
        h, w = frame.shape[:2]

        # Bottom status bar
        bar_h = 36
        cv2.rectangle(out, (0, h-bar_h), (w, h), (15, 15, 15), -1)

        mode_str  = state.get("mode", "VIEW")
        ref_str   = "REF ✓" if state.get("ref_ready") else "NO REF"
        fps_str   = f"FPS {state.get('fps', 0):.1f}"
        match_str = f"Matches {state.get('match_count', 0)}"
        rerr_str  = (f"Reproj {state.get('reproj_err', 0):.1f}px"
                     if state.get("ref_ready") else "")
        nd        = state.get("n_defects", 0)
        verdict   = state.get("verdict", "")
        v_color   = (0, 220, 0) if verdict == "PASS" else \
                    (0, 0, 220) if verdict == "FAIL" else (180, 180, 180)

        bottom = h - 10
        cv2.putText(out, f"[{mode_str}]", (8, bottom),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(out, ref_str, (110, bottom),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if state.get("ref_ready") else (80, 80, 200), 1)
        cv2.putText(out, fps_str, (210, bottom),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        if state.get("ref_ready"):
            cv2.putText(out, match_str, (310, bottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(out, rerr_str, (460, bottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        if verdict:
            vtext = f"{verdict}  ({nd} defect{'s' if nd != 1 else ''})"
            cv2.putText(out, vtext, (w-260, bottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, v_color, 2)

        # Top: key hints
        hints = "R=ref  I=inspect  G=grid  D=diff  S=save  Q=quit"
        cv2.putText(out, hints, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

        return out

    @staticmethod
    def side_by_side(left: np.ndarray,
                     right: np.ndarray, label: str = "") -> np.ndarray:
        """Stack two same-size frames horizontally."""
        h = max(left.shape[0], right.shape[0])
        # Ensure same height
        if left.shape[0] != h:
            left  = cv2.resize(left,  (left.shape[1],  h))
        if right.shape[0] != h:
            right = cv2.resize(right, (right.shape[1], h))
        combined = np.hstack([left, right])
        if label:
            cv2.putText(combined, label, (left.shape[1]+8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        return combined


# ──────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="AOI PCB Inspection System")
    p.add_argument("--ref",        default=None,
                   help="Path to reference image (skips live capture)")
    p.add_argument("--width",      type=int, default=Cfg.OUTPUT_W)
    p.add_argument("--height",     type=int, default=Cfg.OUTPUT_H)
    p.add_argument("--camera-id",  type=int, default=Cfg.CAMERA_ID,
                   dest="camera_id")
    p.add_argument("--no-inspect", action="store_true",
                   help="Start in view-only mode")
    return p.parse_args()


def main():
    args = parse_args()

    cfg           = Cfg()
    cfg.OUTPUT_W  = args.width
    cfg.OUTPUT_H  = args.height
    cfg.CAMERA_ID = args.camera_id

    cfg.SNAPSHOT_DIR.mkdir(exist_ok=True)

    # ── Init pipeline components ──────────────────────────────────────────────
    stream   = VideoStream(cfg)
    ref_mgr  = ReferenceManager(cfg)
    matcher  = FeatureMatcher(cfg)
    aligner  = HomographyAligner(cfg)
    detector = DefectDetector(cfg)
    logger   = CSVLogger(cfg)
    renderer = Renderer(cfg)

    if args.ref:
        ref_mgr.load(args.ref)

    # ── Start stream ──────────────────────────────────────────────────────────
    stream.start()

    # ── Application state ─────────────────────────────────────────────────────
    show_grid    = True
    show_diff    = False
    inspecting   = not args.no_inspect
    running      = True
    frame_n      = 0
    fps_counter  = 0
    fps_val      = 0.0
    fps_t        = time.time()

    match_count  = 0
    reproj_err   = 0.0
    last_defects : list = []
    verdict      = ""

    def shutdown(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("\n[AOI]  Pipeline ready.")
    print("[AOI]  OpenCV window will open when first frame arrives.")
    if not ref_mgr.ready:
        print("[AOI]  Press  R  over a KNOWN-GOOD PCB to set reference.\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while running:

        frame = stream.read(timeout=2.0)
        if frame is None:
            print("[AOI]  No frame — checking pipeline health...")
            continue

        frame_n     += 1
        fps_counter += 1
        if fps_counter >= 30:
            fps_val     = fps_counter / (time.time() - fps_t + 1e-6)
            fps_t       = time.time()
            fps_counter = 0

        display = frame.copy()

        # ── Inspection pipeline (only when ref is ready) ──────────────────
        if inspecting and ref_mgr.ready:
            live_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Stage 2 — Feature matching
            src_pts, dst_pts, match_count = matcher.match(ref_mgr, live_gray)

            if src_pts is not None and match_count >= 4:
                # Stage 3 — Homography alignment
                warped, reproj_err = aligner.align(
                    frame, src_pts, dst_pts, ref_mgr.frame.shape)

                if warped is not None:
                    # Stage 4 — Defect detection
                    last_defects, diff_map = detector.detect(
                        ref_mgr.frame, warped)

                    verdict = "FAIL" if last_defects else "PASS"

                    # Stage 5 — CSV logging (only on new defects)
                    if last_defects:
                        logger.log(last_defects, frame_n)

                    # Draw defects on display
                    display = renderer.draw_defects(display, last_defects)

                    # Optional diff side panel
                    if show_diff:
                        diff_resized = cv2.resize(
                            diff_map, (display.shape[1]//3, display.shape[0]//3))
                        display[0:diff_resized.shape[0],
                                0:diff_resized.shape[1]] = diff_resized

            elif match_count < 4:
                verdict = "ALIGN_FAIL"
                cv2.putText(display,
                            f"⚠  Low matches ({match_count}) — move PCB closer / improve lighting",
                            (10, cfg.OUTPUT_H//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

        # ── Grid ──────────────────────────────────────────────────────────
        if show_grid:
            display = renderer.draw_grid(display)

        # ── HUD ───────────────────────────────────────────────────────────
        display = renderer.draw_hud(display, {
            "mode":        "INSPECT" if inspecting else "VIEW",
            "ref_ready":   ref_mgr.ready,
            "fps":         fps_val,
            "frame_n":     frame_n,
            "match_count": match_count,
            "reproj_err":  reproj_err,
            "n_defects":   len(last_defects),
            "verdict":     verdict if inspecting else "",
        })

        cv2.imshow("AOI System", display)

        # ── Keyboard ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):          # Q or ESC → quit
            running = False

        elif key == ord("r"):              # R → capture reference
            ref_mgr.set(frame)
            last_defects.clear()
            verdict      = ""
            match_count  = 0
            reproj_err   = 0.0
            print(f"[AOI]  Reference captured (frame {frame_n}).")

        elif key == ord("i"):              # I → toggle inspection
            inspecting = not inspecting
            last_defects.clear()
            verdict = ""
            print(f"[AOI]  Inspection {'ON' if inspecting else 'OFF'}.")

        elif key == ord("g"):              # G → toggle grid
            show_grid = not show_grid

        elif key == ord("d"):              # D → toggle diff panel
            show_diff = not show_diff
            if show_diff and not ref_mgr.ready:
                print("[AOI]  Set a reference first (press R).")
                show_diff = False

        elif key == ord("s"):              # S → save snapshot
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = cfg.SNAPSHOT_DIR / f"snapshot_{ts}_f{frame_n}.png"
            cv2.imwrite(str(fname), display)
            print(f"[AOI]  Saved {fname}")

    # ── Shutdown ──────────────────────────────────────────────────────────────
    print(f"\n[AOI]  Session ended — {frame_n} frames processed.")
    if cfg.CSV_PATH.exists():
        print(f"[AOI]  Defect log: {cfg.CSV_PATH}")
    cv2.destroyAllWindows()
    stream.stop()


if __name__ == "__main__":
    main()