# 3. Methodology

## 3.1 System Overview

An automated optical inspection (AOI) system was designed, developed, and tested for real-time detection and classification of printed circuit board (PCB) defects. The system consists of a detection pipeline, a processing framework, a graphical user interface, and a reporting module. The inspection framework combines deep learning-based object detection (YOLOv8n) with classical computer vision techniques (ORB feature matching, homography estimation, and pixel-difference analysis). User commands are provided through a PyQt5-based graphical interface, and detection results are logged to CSV files and exported as PDF/PNG reports.

## 3.2 System Architecture

The AOI system is structured in five integrated subsystems:

**Image Acquisition Subsystem**
- Camera interface using OpenCV (cv2.VideoCapture)
- Support for USB cameras, webcams, and virtual cameras (Camo, DroidCam)
- Frame capture at 30 frames per second (fps) in 1280×720 BGR format

**Detection Pipeline Subsystem**
- YOLO detector: real-time defect classification (always active)
- ORB alignment module: reference-based frame alignment (optional)
- Pixel-difference detector: contextual anomaly detection (optional)
- Hybrid merger: combines detection results with IoU-based suppression

**Quality Assurance Subsystem**
- Temporal defect tracker: confirms detections across multiple frames
- Exposure normalization: compensates for auto-exposure drift
- Morphological filtering: noise reduction and edge preservation

**Data Persistence Subsystem**
- CSV logging: session-based defect recording
- Snapshot capture: annotated frame storage
- Configuration management: centralized parameter tuning

**Reporting and Visualization Subsystem**
- PCB concentration mapping: heat overlay on reference images
- Statistical analysis: defect distribution charts
- PDF/PNG export: multi-page or stacked image reports

The subsystems communicate through Qt signals, ensuring non-blocking operations and responsive user interface behavior.

## 3.3 Detection Pipeline Design

### 3.3.1 YOLO Detection Module

The robotic arm utilizes a YOLOv8n (nano) deep learning model trained on five PCB defect classes. The model was selected for the following reasons:

- **Real-time inference:** Achieves ~20 milliseconds per frame on CPU hardware
- **High accuracy:** Trained model achieves 93.7% mAP50 on custom PCB dataset
- **Generalization:** Works without reference image; appearance-based detection
- **Computational efficiency:** Suitable for CPU-only deployment without GPU requirements

**Defect Classes Detected:**

| Class ID | Defect Name | Severity | Definition |
|----------|------------|----------|-----------|
| 0 | missing_hole | HIGH | Via hole not drilled or blocked by foreign material |
| 1 | mouse_bite | HIGH | Solder mask erosion or copper loss at board edge |
| 2 | open_circuit | HIGH | Broken copper trace or component failure (burnt) |
| 3 | short | HIGH | Unintended solder bridge connecting conductive traces |
| 4 | spur | MEDIUM | Unwanted copper protrusion, whisker, or residual material |

**Model Configuration:**
- Confidence threshold (τ_yolo): 0.80 (configurable, range 0.5–0.95)
- Non-maximum suppression (NMS) IoU: 0.45
- Input resolution: 1280×720 pixels
- Output: Bounding box (x, y, w, h), class label, confidence score

### 3.3.2 ORB Feature Matching and Alignment

When a user captures a reference image of a known-good PCB, the system extracts orientation-invariant ORB (Oriented FAST and Rotated BRIEF) keypoints and binary descriptors. During inspection, live frames are aligned to this reference using homography estimation.

**Feature Extraction Process:**

The feature extraction process involved:

- Extracting up to 5000 ORB keypoints from the live grayscale frame
- Computing binary descriptors for each keypoint
- Performing k-nearest neighbors (k=2) matching with Hamming distance metric
- Applying Lowe's ratio test (λ = 0.75) to filter ambiguous matches
- Outputting matched point correspondences for homography estimation

The ORB detector was selected because of its:

- Rotation invariance (handles board rotation in fixture)
- Computational efficiency (binary descriptors → fast matching)
- Robustness to lighting variation (designed for low-texture environments)
- CPU-only operation (no GPU requirement)

**Homography Estimation and Validation:**

The homography matrix (H) is estimated using RANSAC algorithm with the following validation criteria:

Determinant constraint:
$$\tau_{det,min} < \det(H) < \tau_{det,max}$$

where τ_det,min = 0.05 and τ_det,max = 20.0. This constraint prevents extreme scaling or matrix inversion.

Reprojection error constraint:
$$\text{RMS} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \|\mathbf{p}_i - \mathbf{p}'_i\|^2} < \tau_{reproj}$$

where τ_reproj = 4.0 pixels RMS. This ensures sub-pixel alignment accuracy.

If validation fails, the system falls back to YOLO-only detection without requiring the reference image.

### 3.3.3 Pixel-Difference Detection Module

When a reference image is captured, a pixel-difference detector compares live frames against the reference to identify localized anomalies.

**Preprocessing Step: Exposure Normalization**

A critical preprocessing step normalizes global brightness changes caused by auto-exposure:

```
ref_mean ← mean(reference_grayscale)
live_mean ← mean(live_grayscale_aligned)
scale ← ref_mean / live_mean
live_normalized ← clip(live_grayscale × scale, 0, 255)
```

This normalization cancels global exposure drift while preserving local intensity differences corresponding to physical defects.

**Difference Computation and Thresholding**

The pixel-level difference is computed as:

$$\Delta = |G_{ref} - G_{live,norm}|$$

The difference image is then thresholded at intensity level τ_diff = 40 to create a binary mask:

$$B = \begin{cases} 1 & \text{if } \Delta > \tau_{diff} \\ 0 & \text{otherwise} \end{cases}$$

**Morphological Processing**

Morphological filtering is applied to reduce noise while preserving edge information:

- Morphological opening (3×3 elliptical kernel): removes small noise artifacts
- Morphological closing (7×7 elliptical kernel): fills small gaps within detected regions
- Structuring element: Elliptical kernel (rotationally symmetric, robust to orientation)

**Contour Detection and Filtering**

External contours are extracted from the processed binary image. Contours are filtered based on:

- Area threshold: A_min = 200 pixels²
- Spatial exclusion zones: regions like Camo watermark (bottom-right corner)

### 3.3.4 Heuristic Defect Classification

For each detected contour, the system computes morphological and intensity features and applies heuristic classification rules.

**Feature Computation:**

- Aspect ratio: r = width / height
- Reference region mean intensity: μ_ref
- Live region mean intensity: μ_live  
- Intensity delta: Δμ = μ_live − μ_ref

**Classification Rules:**

The classification process follows the logic below:

If Δμ > 25 AND r ∈ [0.28, 3.5]:
- Label: spurious_copper (extra copper on board)
- Severity: MEDIUM
- Confidence: min(1.0, Δμ / 60)

Else if r > 3.5 OR r < 0.28:
- Label: short (thin solder bridge)
- Severity: HIGH
- Confidence: min(1.0, A / 1500)

Else if Δμ < −20:
- Label: missing_hole (hole blocked or missing component)
- Severity: HIGH
- Confidence: min(1.0, |Δμ| / 80)

Else if μ_live < 35 AND μ_ref < 35:
- Label: open_circuit (burnt trace, very dark region)
- Severity: HIGH
- Confidence: 0.75

Else:
- Label: anomaly (unclassified change)
- Severity: LOW
- Confidence: 0.50

**Confidence Gating:**

Detections with confidence below τ_diff_conf = 0.55 are discarded to reduce noise-induced false positives from the heuristic classifier.

### 3.3.5 Detection Fusion and Suppression

After YOLO and pixel-difference detectors run independently, their results are merged and overlapping detections are suppressed:

Merged detections ← YOLO detections

For each diff detection:
- Compute maximum IoU overlap with YOLO detections
- If overlap < τ_merge (0.30): append to merged list
- If overlap ≥ τ_merge: suppress (YOLO is authoritative)

This suppression prevents double-counting of the same physical defect detected by both pathways.

## 3.4 Temporal Defect Tracking

### 3.4.1 Temporal Confirmation Mechanism

A temporal defect tracker smooths detection results across multiple frames, reducing flicker caused by camera noise and alignment jitter.

**Algorithm:**

The tracker maintains a list of active defect tracks. Each frame, the following sequence occurs:

1. For each existing track (defect, frame_count):
   - Find best-matching detection in current frame (same label, IoU ≥ 0.25)
   - If match found: increment frame_count, update position
   - If no match: delete track (defect disappeared)

2. For each unmatched detection in current frame:
   - Create new track with frame_count = 1

3. Output defects where frame_count ≥ τ_temporal (4 frames)

**Operational Modes:**

Two operational modes are defined based on reference availability:

**YOLO-Only Mode (No Reference):**
- Temporal filtering is disabled
- Defects are reported immediately upon detection
- Response time: 50–70 milliseconds
- Suitable for rapid screening applications

**Hybrid Mode (With Reference):**
- Temporal filtering is enabled
- Defects confirmed across 4 consecutive frames before reporting
- Confirmation delay: ~133 milliseconds (4 frames / 30 fps)
- Noise reduction: ~66% reduction in false positives

## 3.5 System Configuration

The AOI system uses a centralized configuration file (config.py) containing all tunable parameters:

**YOLO Configuration:**
- YOLO_CONF: detection confidence threshold (default: 0.80, range: 0.5–0.95)
- YOLO_IOU: NMS suppression IoU (default: 0.45)

**ORB and Alignment Configuration:**
- ORB_FEATURES: keypoints per frame (default: 5000, range: 1000–10000)
- ORB_MIN_MATCHES: minimum matches for alignment (default: 30)
- LOWE_RATIO: ratio test threshold (default: 0.75, range: 0.5–0.9)
- H_DET_MIN: minimum homography determinant (default: 0.05)
- H_DET_MAX: maximum homography determinant (default: 20.0)
- H_REPROJ_MAX: maximum reprojection error (default: 4.0 px)

**Pixel-Difference Configuration:**
- DIFF_BLUR: Gaussian blur radius (default: 5 px)
- DIFF_THRESH: intensity threshold for binary mask (default: 40, range: 0–255)
- MORPH_OPEN: morphological opening kernel size (default: 3 px)
- MORPH_CLOSE: morphological closing kernel size (default: 7 px)
- MIN_DEFECT_AREA: minimum contour area (default: 200 px²)
- DIFF_MIN_CONF: minimum diff detection confidence (default: 0.55, range: 0.3–0.8)

**Temporal Configuration:**
- TEMPORAL_FRAMES: frames required for confirmation (default: 4, range: 1–10)
- MERGE_IOU_THRESH: YOLO/diff overlap threshold (default: 0.30)

## 3.6 Data Logging and Persistence

### 3.6.1 CSV Logging

Confirmed defects are logged to a timestamped CSV file located in `data/logs/`:

**File naming:** session_YYYYMMDD_HHMMSS.csv

**Logged fields:**
- timestamp: ISO 8601 format (e.g., 2026-06-02T10:00:05)
- frame: frame number in inspection session
- label: defect class name
- severity: HIGH, MEDIUM, or LOW
- x, y, w, h: bounding box coordinates and dimensions (pixels)
- area: contour area (pixels²)
- confidence: detection confidence (0.0–1.0)
- source: detection source (yolo or diff)

### 3.6.2 Snapshot Capture

The system supports manual snapshot capture of annotated frames:

**File location:** `data/snapshots/`

**File naming:** snapshot_YYYYMMDD_HHMMSS.png

**Content:** Full-resolution annotated frame with bounding boxes, class labels, confidence scores, and defect severity color coding.

## 3.7 Report Generation

### 3.7.1 PCB Concentration Mapping

The reporting system generates PCB concentration maps overlaying defect heat on the reference image.

**Algorithm:**

The concentration mapping process involved:

1. Building a 2D heat accumulation grid:
   - For each logged defect, compute centroid (cx, cy)
   - Assign weight based on severity (HIGH=3, MEDIUM=2, LOW=1)
   - Accumulate weight at grid location: heat[cy, cx] += weight

2. Applying Gaussian smoothing:
   - Convolve heat grid with Gaussian kernel (σ = 35 pixels)
   - Normalize to [0, 1] range

3. Compositing visual layers:
   - Background: reference PCB image (user-captured during setup)
   - Overlay: YlOrRd colormap with alpha blending (α = heat_intensity × 0.60)
   - Markers: severity-colored circles at each defect centroid
   - Legend: color code (red=HIGH, orange=MEDIUM, green=LOW)

**Fallback Strategy:**

When no reference image is available, the system generates a plain grid-based heatmap:
- Frame divided into uniform grid cells (64×64 pixels)
- Each cell colored based on accumulated defect count
- Grid boundaries displayed with contour lines

### 3.7.2 Statistical Reporting

The report includes two statistical visualizations:

**Bar Chart:** Defect counts by class
- Horizontal bar chart with class labels
- Bar color corresponds to most common severity in that class
- X-axis: defect count; Y-axis: defect class names

**Pie Chart:** Severity distribution
- Pie chart showing proportion of HIGH, MEDIUM, and LOW severity defects
- Slices colored by severity (red, orange, green)
- Percentage labels for each slice

### 3.7.3 Summary Page

A summary text page displays:

- Total defect count
- Breakdown by severity (HIGH, MEDIUM, LOW)
- Overall verdict: PASS (zero HIGH defects), WARN (HIGH=0 but MEDIUM>0), or FAIL (HIGH>0)
- Verdict displayed in large text with color coding

### 3.7.4 Report Formats

Two export formats are supported:

**PDF Export:**
- Multi-page PDF document
- Page 1: concentration map
- Page 2: statistical charts
- Page 3: summary with verdict

**PNG Export:**
- Single stacked image combining all visualizations
- Vertical stacking with uniform width
- Suitable for rapid preview or email transmission

## 3.8 Software Architecture

### 3.8.1 Component Organization

The software is organized in the following modules:

**Core Detection Pipeline (core/):**
- camera.py: camera interface and device discovery
- reference.py: reference image management and feature extraction
- aligner.py: ORB matching and homography estimation
- detector.py: YOLO, diff, and hybrid detectors
- pipeline.py: main QThread orchestrating detection loop
- logger.py: CSV logging functionality

**User Interface (ui/):**
- main_window.py: root Qt window and signal wiring
- widgets/: video display, camera selection, control buttons, defect table
- dialogs/: report generation and export dialog

**Report Generation (reports/):**
- generator.py: PDF/PNG export pipeline
- heatmap.py: concentration map and grid heatmap rendering

### 3.8.2 Communication Protocols

**Qt Signals:**
- Pipeline emits frame_ready signal to update video display
- Pipeline emits defects_ready signal to update defect panel
- Buttons emit signals triggering corresponding actions

**Serial Communication:**
- No hardware serial interface required
- USB camera interface handled by OpenCV
- All communication internal to application

### 3.8.3 Threading Model

**Main Thread (Qt Event Loop):**
- Handles user input (button clicks, keyboard)
- Updates UI widgets
- Manages dialogs

**Worker Thread (InspectionPipeline QThread):**
- Runs detection pipeline continuously
- Does not block main thread
- Emits signals to update UI asynchronously

## 3.9 System Integration and Testing

After software development and hardware setup, the system was integrated and tested to verify overall performance and reliability.

**Testing Objectives:**

The testing phase focused on:

- Detection accuracy (precision, recall, F1-score)
- Speed performance (inference latency, throughput)
- Robustness to environmental variations (lighting, camera jitter)
- Communication reliability (serial communication, signal integrity)
- Software stability (crash testing, edge cases)
- Gripper and mechanical operation (range of motion, accuracy)

**Test Procedures:**

The test procedures involved:

1. Unit testing: individual modules (YOLO, ORB, diff detector)
2. Integration testing: interaction between detection modules
3. System testing: end-to-end workflow (capture → detect → report)
4. Robustness testing: poor lighting, texture-less boards, alignment failures
5. Stress testing: sustained 30 fps operation for extended periods
6. User acceptance testing: teleoperation responsiveness and ease of use

**Performance Metrics Evaluated:**

- Precision: TP / (TP + FP) — fraction of reported defects that are true positives
- Recall: TP / (TP + FN) — fraction of actual defects detected
- F1-score: 2×(Precision × Recall) / (Precision + Recall)
- Latency: time from detection to report output
- Throughput: defects processed per second
- Reliability: percentage of frames processed without error

## 3.10 Operational Modes

The system supports two distinct operational modes allowing users to choose between speed and accuracy:

**Mode 1: YOLO-Only Screening**
- No reference image required
- Click "Start Inspection" immediately after camera connection
- YOLO detection runs without temporal filtering
- Results appear in real-time defect panel
- Report uses grid-based heatmap (no PCB context)
- Use case: rapid first-pass screening, quality gating

**Mode 2: Hybrid Reference-Based Inspection**
- User captures reference image of known-good PCB
- ORB alignment and pixel-difference detection enabled
- Temporal smoothing confirms detections across 4 frames
- Enhanced accuracy with noise reduction
- Report includes PCB concentration map on reference image
- Use case: detailed defect analysis, quality documentation

## 3.11 Performance Characteristics

**Computational Performance:**

| Component | Time (ms) | Hardware | Notes |
|-----------|-----------|----------|-------|
| YOLO inference | ~20 | CPU | YOLOv8n, 1280×720 input |
| ORB detection & matching | ~10 | CPU | 5000 features, KNN matching |
| Homography estimation | ~2 | CPU | RANSAC, small point sets |
| Morphological operations | ~10 | CPU | Filtering, contour detection |
| Temporal tracking | ~1 | CPU | IoU matching |
| **Total detection** | **~50** | **CPU** | **Per-frame latency** |
| Display rendering (Qt) | ~5 | GPU/CPU | Every 2nd frame rendered |

**Throughput:**
- Raw detection: 30 fps
- Display update: 15 fps (DISPLAY_INTERVAL=2 skips frames)
- Temporal confirmation: 4-frame delay (~133 ms)
- Total end-to-end latency: ~200 ms

**Memory Usage:**
- YOLO model: ~150 MB
- Feature storage: ~5–10 MB per frame
- Frame buffers: ~10 MB
- Total: ~170–180 MB

---

**This methodology document is formatted for academic presentation, technical reports, and peer review.**
