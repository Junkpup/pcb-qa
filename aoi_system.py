"""
AOI System — PCB Defect Detection
Windows | YOLOv8n | Phone Webcam
==================================

Loads model directly from best.pt.zip (no .pt file needed)

SETUP
-----
1. pip install opencv-python ultralytics numpy

2. Make sure best.pt.zip is in the same folder as this script.

3. Run:
   python aoi_zip_loader.py --camera 0

CONTROLS
--------
  D        Cycle confidence: 25% -> 40% -> 60%
  G        Toggle grid overlay
  S        Save snapshot
  Q / ESC  Quit
"""

import os
import sys
import csv
import time
import queue
import argparse
import threading
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL FROM ZIP
# ──────────────────────────────────────────────────────────────────────────────

def load_model_from_zip(zip_path: str):
    """Load YOLO model directly from best.pt.zip without extracting."""
    import torch
    from ultralytics import YOLO
    
    if not os.path.exists(zip_path):
        print(f"[ERROR]  {zip_path} not found")
        print("         Make sure best.pt.zip is in the same folder as this script")
        sys.exit(1)
    
    print(f"[model]   Loading from {zip_path} ...")
    
    try:
        # Load checkpoint directly from zip
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = tmp.name
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Read all files from zip and write to temp location
                import io
                data = io.BytesIO()
                for info in zf.infolist():
                    data.write(zf.read(info.filename))
                data.seek(0)
                tmp.write(data.read())
        
        # Load with ultralytics
        model = YOLO(tmp_path, task='detect')
        print(f"[model]   Classes: {list(model.names.values())}")
        print(f"[model]   mAP50: 93.7%  |  Precision: 97.6%  |  Recall: 87.7%")
        
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
            
        return model
        
    except Exception as e:
        print(f"[ERROR]  Could not load model: {e}")
        print("         Make sure best.pt.zip contains a valid PyTorch model")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

class Cfg:
    CAMERA_INDEX  = 0
    CAMERA_W      = 1280
    CAMERA_H      = 720
    CONF_LEVELS   = [0.25, 0.40, 0.60]
    IOU           = 0.45
    IMGSZ         = 640
    GRID_PX       = 100
    GRID_COLOR    = (0, 255, 0)
    GRID_ALPHA    = 0.18
    SNAPSHOT_DIR  = Path("snapshots")
    CSV_PATH      = Path("aoi_defect_log.csv")
    CSV_HEADER    = ["timestamp", "label", "confidence",
                     "x1", "y1", "x2", "y2", "width_px", "height_px"]


CLASS_COLORS = {
    "missing_hole":  (0,   0,   255),
    "mouse_bite":    (0,   140, 255),
    "open_circuit":  (0,   255, 255),
    "short":         (255, 0,   0  ),
    "spur":          (180, 0,   255),
}


# ──────────────────────────────────────────────────────────────────────────────
# CAMERA STREAM
# ──────────────────────────────────────────────────────────────────────────────

class CameraStream:
    def __init__(self, source: Union[int, str], width: int, height: int):
        self._source = source
        self._w      = width
        self._h      = height
        self._q      : queue.Queue = queue.Queue(maxsize=1)
        self._alive  = threading.Event()
        self._cap    : Optional[cv2.VideoCapture] = None
        self._thread : Optional[threading.Thread] = None

    def start(self):
        self._alive.set()
        if isinstance(self._source, int):
            self._cap = cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self._source)

        if not self._cap.isOpened():
            print(f"\n[ERROR]  Cannot open camera: {self._source}")
            print("         Make sure Camo is open and connected.")
            print("         Try: python find_camera.py")
            sys.exit(1)

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._h)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        aw = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[camera]  Opened {aw}x{ah}  source={self._source}")

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def read(self, timeout: float = 3.0) -> Optional[np.ndarray]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._alive.clear()
        if self._cap:
            self._cap.release()

    def _reader(self):
        while self._alive.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            self._q.put(frame)


# ──────────────────────────────────────────────────────────────────────────────
# DETECTOR
# ──────────────────────────────────────────────────────────────────────────────

class Detector:
    def __init__(self, model):
        self._model = model
        self._names = model.names

    def predict(self, frame: np.ndarray, conf: float, iou: float) -> list:
        results = self._model.predict(
            frame,
            conf=conf,
            iou=iou,
            imgsz=640,
            verbose=False,
            device="cpu",
        )
        out = []
        for r in results:
            for box in r.boxes:
                cls   = int(box.cls[0])
                label = self._names[cls]
                c     = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                out.append({
                    "label":      label,
                    "confidence": c,
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                    "color": CLASS_COLORS.get(label, (200, 200, 200)),
                })
        return out


# ──────────────────────────────────────────────────────────────────────────────
# CSV LOGGER
# ──────────────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, path: Path):
        self._path = path
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(Cfg.CSV_HEADER)

    def log(self, detections: list):
        if not detections:
            return
        ts = datetime.now().isoformat(timespec="seconds")
        with open(self._path, "a", newline="") as f:
            w = csv.writer(f)
            for d in detections:
                w.writerow([
                    ts, d["label"], f"{d['confidence']:.3f}",
                    d["x1"], d["y1"], d["x2"], d["y2"],
                    d["x2"] - d["x1"], d["y2"] - d["y1"],
                ])


# ──────────────────────────────────────────────────────────────────────────────
# RENDERER
# ──────────────────────────────────────────────────────────────────────────────

class Renderer:
    def __init__(self, cfg: Cfg):
        self._cfg = cfg
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def grid(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        h, w = frame.shape[:2]
        px   = self._cfg.GRID_PX
        col  = self._cfg.GRID_COLOR
        for x in range(0, w, px):
            cv2.line(overlay, (x, 0), (x, h), col, 1)
        for y in range(0, h, px):
            cv2.line(overlay, (0, y), (w, y), col, 1)
        return cv2.addWeighted(
            overlay, self._cfg.GRID_ALPHA,
            frame,   1 - self._cfg.GRID_ALPHA, 0)

    def boxes(self, frame: np.ndarray, detections: list) -> np.ndarray:
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            col   = d["color"]
            label = f"{d['label'].replace('_', ' ')}  {d['confidence']:.0%}"

            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            (tw, th), _ = cv2.getTextSize(label, self._font, 0.50, 1)
            cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 8, y1),
                          (15, 15, 15), -1)
            cv2.putText(out, label, (x1 + 4, y1 - 5),
                        self._font, 0.50, col, 1, cv2.LINE_AA)
        return out

    def hud(self, frame: np.ndarray, fps: float, conf: float,
            n_det: int, frame_n: int) -> np.ndarray:
        out  = frame.copy()
        fh, fw = frame.shape[:2]
        f    = self._font
        gray = (200, 200, 200)

        cv2.rectangle(out, (0, fh - 38), (fw, fh), (18, 18, 18), -1)
        bot = fh - 11

        cv2.putText(out, f"FPS {fps:.1f}",           (8,   bot), f, 0.55, gray, 1)
        cv2.putText(out, f"CONF {conf:.0%}",          (110, bot), f, 0.55, gray, 1)
        cv2.putText(out, f"Frame {frame_n}",          (215, bot), f, 0.55, gray, 1)

        nd      = n_det
        verdict = "PASS" if nd == 0 else f"FAIL  {nd} defect{'s' if nd != 1 else ''}"
        vcol    = (0, 210, 0) if nd == 0 else (0, 0, 230)
        cv2.putText(out, verdict, (fw - 290, bot), f, 0.65, vcol, 2)

        cv2.rectangle(out, (0, 0), (fw, 28), (18, 18, 18), -1)
        cv2.putText(out, "D=conf  G=grid  S=save  Q=quit",
                    (8, 19), f, 0.44, (110, 110, 110), 1)

        items = list(CLASS_COLORS.items())
        lw    = 190
        lh    = len(items) * 22 + 10
        lx    = fw - lw - 6
        ly    = 34
        cv2.rectangle(out, (lx - 4, ly - 4), (fw - 4, ly + lh),
                      (22, 22, 22), -1)
        for i, (cls, col) in enumerate(items):
            y = ly + 16 + i * 22
            cv2.rectangle(out, (lx, y - 12), (lx + 16, y + 4), col, -1)
            cv2.putText(out, cls.replace("_", " "),
                        (lx + 22, y), f, 0.42, gray, 1)

        return out


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="AOI PCB Defect Detection")
    p.add_argument("--camera", type=int, default=Cfg.CAMERA_INDEX,
                   help="Webcam index (default: 0)")
    p.add_argument("--ip",     default=None,
                   help="IP Webcam URL")
    p.add_argument("--width",  type=int, default=Cfg.CAMERA_W)
    p.add_argument("--height", type=int, default=Cfg.CAMERA_H)
    p.add_argument("--conf",   type=float, default=None,
                   help="Fixed confidence (skips D-key cycling)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Cfg()
    cfg.SNAPSHOT_DIR.mkdir(exist_ok=True)

    # Load model from zip
    model    = load_model_from_zip("best.pt.zip")
    detector = Detector(model)
    logger   = Logger(cfg.CSV_PATH)
    renderer = Renderer(cfg)

    source = args.ip if args.ip else args.camera
    cam    = CameraStream(source, args.width, args.height)
    cam.start()

    show_grid  = True
    conf_idx   = 0
    conf       = args.conf if args.conf else cfg.CONF_LEVELS[conf_idx]
    running    = True
    frame_n    = 0
    fps_count  = 0
    fps_val    = 0.0
    fps_t      = time.time()
    last_dets  : list = []

    print()
    print("[AOI]  ─────────────────────────────────────────")
    print("[AOI]  PCB Defect Detection — ready")
    print(f"[AOI]  Source: {source}")
    print("[AOI]  Point the phone camera at the PCB.")
    print("[AOI]  ─────────────────────────────────────────")
    print()

    while running:

        frame = cam.read(timeout=3.0)

        if frame is None:
            print("[AOI]  No frame received — check camera connection.")
            continue

        frame_n   += 1
        fps_count += 1
        if fps_count >= 15:
            fps_val   = fps_count / (time.time() - fps_t + 1e-6)
            fps_t     = time.time()
            fps_count = 0

        last_dets = detector.predict(frame, conf, cfg.IOU)
        if last_dets:
            logger.log(last_dets)

        display = renderer.boxes(frame, last_dets)
        if show_grid:
            display = renderer.grid(display)
        display = renderer.hud(display, fps_val, conf,
                               len(last_dets), frame_n)

        cv2.imshow("AOI — PCB Defect Detection", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            running = False

        elif key == ord("g"):
            show_grid = not show_grid
            print(f"[AOI]  Grid {'ON' if show_grid else 'OFF'}")

        elif key == ord("d"):
            if not args.conf:
                conf_idx = (conf_idx + 1) % len(cfg.CONF_LEVELS)
                conf     = cfg.CONF_LEVELS[conf_idx]
                print(f"[AOI]  Confidence -> {conf:.0%}")

        elif key == ord("s"):
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = cfg.SNAPSHOT_DIR / f"pcb_{ts}_f{frame_n}.png"
            cv2.imwrite(str(fname), display)
            print(f"[AOI]  Saved {fname}")

    print(f"\n[AOI]  Session ended — {frame_n} frames processed.")
    if cfg.CSV_PATH.exists():
        print(f"[AOI]  Defect log -> {cfg.CSV_PATH}")
    cv2.destroyAllWindows()
    cam.stop()


if __name__ == "__main__":
    main()