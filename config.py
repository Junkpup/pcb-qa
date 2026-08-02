
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Cfg:
    # ── Camera / video ────────────────────────────────────────────────────────
    CAMERA_INDEX: int = 0          # cv2.VideoCapture index (Camo / capture card)
    OUTPUT_W: int = 1280
    OUTPUT_H: int = 720
    TARGET_FPS: int = 30

    # ── YOLO detector ─────────────────────────────────────────────────────────
    # CoreML model (.mlpackage) is used automatically on macOS if it exists —
    # run scripts/export_coreml.py once to generate it (5–10x faster on CPU).
    MODEL_PATH: Path = field(default_factory=lambda: Path("models/best.pt"))
    YOLO_CONF: float = 0.80        # detection confidence threshold — raise if too many false positives
    YOLO_IOU: float = 0.45         # NMS IoU threshold

    # ── Display performance ───────────────────────────────────────────────────
    # Detection runs every frame; UI is updated every Nth frame to reduce
    # rendering overhead on older hardware (2 = ~15 fps display, full detection).
    DISPLAY_INTERVAL: int = 2

    # ── Exclusion zones (Camo watermark / irrelevant frame regions) ───────────
    # Each entry is (x1_frac, y1_frac, x2_frac, y2_frac) in 0–1 normalised coords.
    # Default covers the Camo logo at the bottom-right corner of the frame.
    EXCLUSION_ZONES: tuple = ((0.82, 0.88, 1.0, 1.0),)

    # ── Known YOLO class names (read from model at startup; do not edit) ──────
    # Actual classes in models/best.pt:
    #   0: missing_hole  1: mouse_bite  2: open_circuit  3: short  4: spur
    # 'spurious_copper' is NOT in the model — detected by the diff pipeline only.
    YOLO_CLASS_NAMES: tuple = (
        "missing_hole", "mouse_bite", "open_circuit", "short", "spur"
    )

    GRID_PX: int = 100
    GRID_COLOR: tuple = (0, 255, 0)   # BGR green
    GRID_ALPHA: float = 0.20

    ORB_FEATURES: int = 5000
    ORB_MIN_MATCHES: int = 30
    LOWE_RATIO: float = 0.75

    H_DET_MIN: float = 0.05
    H_DET_MAX: float = 20.0
    H_REPROJ_MAX: float = 4.0      # px RMS ceiling

    DIFF_BLUR: int = 5
    DIFF_THRESH: int = 40
    MORPH_OPEN: int = 3
    MORPH_CLOSE: int = 7
    MIN_DEFECT_AREA: int = 200     # px²
    DIFF_MIN_CONF: float = 0.55    # discard diff detections below this confidence

    TEMPORAL_FRAMES: int = 4

    MERGE_IOU_THRESH: float = 0.30

    SNAPSHOT_DIR: Path = Path("data/snapshots")
    REFERENCE_DIR: Path = Path("data/references")
    LOG_DIR: Path = Path("data/logs")

    
    REPORT_GRID_PX: int = 64      

    def ensure_dirs(self) -> None:
        for p in (self.SNAPSHOT_DIR, self.REFERENCE_DIR, self.LOG_DIR):
            Path(p).mkdir(parents=True, exist_ok=True)
