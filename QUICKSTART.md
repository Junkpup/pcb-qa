# Quick Start & Reference Guide

## Installation & Setup

### Prerequisites
- Python 3.12+
- Windows 10/11
- USB camera or Camo virtual camera

### Installation

```bash
cd "C:\Users\arnav\OneDrive\Documents\Work\4th Sem EL\qa_fixed"
pip install -r requirements.txt
```

### First Run

```bash
python main.py
```

---

## Usage Workflow

### Option 1: YOLO-Only Detection (No Reference)

**Best for:** Fast screening, checking if defects exist.

1. **Connect camera**
   - Dropdown → select camera (Camo, USB, etc.)
   - Click "Connect"
   - HUD shows "NO REF" in blue

2. **Start inspection**
   - Click "Start Inspection"
   - YOLO detections appear instantly in the panel
   - No lag, no temporal smoothing

3. **Generate report**
   - Click "Generate Report"
   - Exports PDF/PNG with grid heatmap
   - Can sort defects in preview

### Option 2: Hybrid Detection (With Reference)

**Best for:** High accuracy, contextual defect detection, PCB concentration map.

1. **Connect camera**
   - Dropdown → select camera
   - Click "Connect"

2. **Capture reference**
   - Point at a **known-good PCB**
   - Click "Capture Reference"
   - Wait for alignment to stabilise (~1 second)
   - HUD shows "REF ✓" in green

3. **Start inspection**
   - Click "Start Inspection"
   - Now inspecting defective boards
   - YOLO + diff detection run together
   - Temporal smoothing filters noise (4-frame delay, imperceptible)

4. **Generate report**
   - Click "Generate Report"
   - Exports PDF with **PCB concentration map** overlaid on your reference photo
   - Higher-quality visualisation

---

## Common Tasks

### Adjust Detection Sensitivity

Edit `config.py`:

```python
# Make YOLO stricter (fewer false positives)
YOLO_CONF = 0.90  # was 0.80

# Make YOLO looser (catch more defects)
YOLO_CONF = 0.65  # was 0.80

# Reduce diff noise
DIFF_THRESH = 50  # was 40
DIFF_MIN_CONF = 0.65  # was 0.55

# Faster response (less smoothing)
TEMPORAL_FRAMES = 2  # was 4
```

Restart `python main.py` after editing.

### Improve Alignment (ORB Matching)

If you see "ALIGN_FAIL" (not enough keypoints):

```python
# Boost feature detection
ORB_FEATURES = 8000  # was 5000 (slower, more robust)
ORB_MIN_MATCHES = 20  # was 30 (looser threshold)

# Relax homography validation
H_REPROJ_MAX = 6.0  # was 4.0
```

**OR improve lighting:**
- Add ring light around camera
- Use matte PCBs (glossy surfaces cause reflections)
- Avoid shadows/backlighting

### Fix "Everything is Missing Hole"

Symptom: All diff detections classified as `missing_hole`.

**Root cause:** Auto-exposure drift → live frame darker than reference.

**Fix:** Already handled by exposure normalisation in `core/detector.py`.  
If still occurs, increase threshold:

```python
DIFF_THRESH = 50  # was 40
```

### Save Session Data

Defects automatically logged to:
- **CSV:** `data/logs/session_20260602_100000.csv`
- **Snapshots:** `data/snapshots/snapshot_20260602_100000.png`
- **Report:** Export from the "Generate Report" dialog

---

## Keyboard Shortcuts (HUD Hint)

Displayed at top of live video:

| Key | Action |
|-----|--------|
| R | Capture Reference |
| I | Toggle Inspection |
| G | Toggle Grid overlay |
| D | Toggle Diff heatmap (if ref available) |
| S | Save snapshot |
| Q | Quit |

(Currently UI-only; keyboard hooks not yet implemented.)

---

## Troubleshooting

### Camera Not Detected

1. Check camera is connected (USB / Camo)
2. Click "Refresh" in camera panel
3. If using Camo: ensure Camo app running on phone + USB enabled
4. If using DroidCam: same as above

### PyTorch DLL Error

If you see `OSError [WinError 1114]` on startup:

```bash
python -m pip uninstall torch -y
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Restart:
```bash
python main.py
```

### Defects Flicker (Change Every Frame)

With reference captured, temporal smoothing is active (4-frame delay).  
If still flickering:

```python
TEMPORAL_FRAMES = 6  # increase smoothing
DIFF_MIN_CONF = 0.70  # increase diff confidence gate
```

### Report Export Fails

1. Ensure `data/logs/` directory exists (created auto on first detection)
2. Ensure you have at least one detected defect logged
3. Check disk space (PDF can be 1–5 MB per session)

### No Defects Detected

1. **YOLO-only mode:** Check YOLO_CONF is not too high
   ```python
   YOLO_CONF = 0.70  # more lenient
   ```

2. **Hybrid mode:** Ensure reference captured (HUD shows "REF ✓")
   - If "NO REF" shown, click "Capture Reference"

3. **Check model:** Verify `models/best.pt` exists and is ~6.2 MB
   ```bash
   dir models\
   ```

4. **Lighting:** Ensure adequate illumination

---

## Configuration Reference

**File:** `config.py`

### Camera Settings

```python
CAMERA_INDEX = 0              # Camera index (auto-selected in dropdown)
OUTPUT_W = 1280               # Frame width (pixels)
OUTPUT_H = 720                # Frame height (pixels)
TARGET_FPS = 30               # Target frame rate
DISPLAY_INTERVAL = 2          # Skip frames for rendering (1=every frame, higher=faster)
```

### YOLO Settings

```python
MODEL_PATH = Path("models/best.pt")
YOLO_CONF = 0.80              # **Confidence threshold (0.0–1.0)**
YOLO_IOU = 0.45               # NMS IoU threshold
```

### ORB & Alignment

```python
ORB_FEATURES = 5000           # Keypoints (higher=more robust, slower)
ORB_MIN_MATCHES = 30          # Min matches to attempt alignment
LOWE_RATIO = 0.75             # Ratio test (lower=stricter)
H_DET_MIN = 0.05              # Min homography determinant
H_DET_MAX = 20.0              # Max homography determinant
H_REPROJ_MAX = 4.0            # Max reprojection error (px)
```

### Diff Detection

```python
DIFF_BLUR = 5                 # Gaussian blur radius
DIFF_THRESH = 40              # **Intensity threshold (0–255)**
MORPH_OPEN = 3                # Morphological opening size (px)
MORPH_CLOSE = 7               # Morphological closing size (px)
MIN_DEFECT_AREA = 200         # Min blob size (px²)
DIFF_MIN_CONF = 0.55          # **Min confidence to report diff detection**
```

### Temporal Smoothing

```python
TEMPORAL_FRAMES = 4           # **Frames to confirm (higher=more stable)**
MERGE_IOU_THRESH = 0.30       # YOLO/diff overlap to suppress
```

### I/O Paths

```python
SNAPSHOT_DIR = Path("data/snapshots")
REFERENCE_DIR = Path("data/references")
LOG_DIR = Path("data/logs")
```

**Bold parameters** are the most common tuning knobs.

---

## Performance Tips

| Goal | Adjustment |
|------|------------|
| Faster detection | ↓ ORB_FEATURES; ↑ DISPLAY_INTERVAL |
| More accurate | ↑ ORB_FEATURES; ↓ YOLO_CONF; ↑ TEMPORAL_FRAMES |
| Better alignment | ↑ ORB_FEATURES; ↓ H_REPROJ_MAX |
| Fewer false positives | ↑ YOLO_CONF; ↑ DIFF_THRESH; ↑ DIFF_MIN_CONF |
| More detections | ↓ YOLO_CONF; ↓ DIFF_THRESH; ↓ DIFF_MIN_CONF |

---

## File Structure

```
qa_fixed/
├── main.py                          # Entry point
├── config.py                        # Configuration dataclass
├── METHODOLOGY.md                   # This file
├── requirements.txt                 # Dependencies
│
├── core/                            # Detection pipeline
│   ├── camera.py                    # Camera abstraction
│   ├── reference.py                 # Reference manager
│   ├── aligner.py                   # ORB + homography
│   ├── detector.py                  # YOLO + diff + hybrid
│   ├── pipeline.py                  # QThread orchestration
│   └── logger.py                    # CSV logging
│
├── ui/                              # PyQt5 GUI
│   ├── main_window.py               # Root window
│   ├── widgets/                     # Sub-widgets
│   │   ├── video_widget.py          # Live feed display
│   │   ├── camera_panel.py          # Camera selection
│   │   ├── controls_bar.py          # Buttons + status
│   │   └── defect_panel.py          # Defect table
│   └── dialogs/
│       └── report_dialog.py         # Report export
│
├── reports/                         # Report generation
│   ├── generator.py                 # PDF/PNG export
│   ├── heatmap.py                   # Concentration maps
│   └── __init__.py
│
├── models/
│   ├── best.pt                      # YOLOv8n trained model
│   └── best.mlpackage/              # CoreML (macOS only)
│
├── data/
│   ├── references/                  # Saved reference images
│   ├── snapshots/                   # Manual captures
│   └── logs/                        # Session CSV logs
│
└── _archive/                        # Legacy versions
```

---

## Support & Reference

- **Model training:** See `scripts/export_coreml.py` (macOS CoreML export)
- **Defect definitions:** See `core/detector.py:_CLASS_SEVERITY`
- **Report customisation:** See `reports/generator.py` and `reports/heatmap.py`
- **UI theming:** Stylesheet in `main.py` (dark palette colours)

---

**Last updated:** June 2026 | **Version:** 1.0
