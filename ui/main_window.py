"""
ui/main_window.py — Root QMainWindow for the AOI PCB Inspection System.

Layout
------
Left    : CameraPanel (device selection)
Centre  : VideoWidget (live annotated feed)
Right   : DefectPanel (live defect table)
Bottom  : ControlsBar (toolbar + status strip)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QMessageBox,
)

from config import Cfg
from core.camera import CameraSource
from core.pipeline import InspectionPipeline
from core.detector import Defect
from ui.widgets.video_widget import VideoWidget
from ui.widgets.camera_panel import CameraPanel
from ui.widgets.defect_panel import DefectPanel
from ui.widgets.controls_bar import ControlsBar
from ui.dialogs.report_dialog import ReportDialog


class MainWindow(QMainWindow):
    """Root application window."""

    def __init__(self, cfg: Cfg):
        super().__init__()
        self._cfg      = cfg
        self._camera   = CameraSource(cfg)
        self._pipeline: InspectionPipeline | None = None
        self._last_frame: np.ndarray | None = None

        self.setWindowTitle("AOI PCB Inspection System")
        self.setMinimumSize(1280, 780)
        self.setStyleSheet("background:#1e1e1e; color:#e0e0e0;")

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Main horizontal splitter ──────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet("QSplitter::handle { background:#333; }")

        self._cam_panel    = CameraPanel(self._cfg, self._camera)
        self._video_widget = VideoWidget()
        self._defect_panel = DefectPanel()

        self._cam_panel.setMaximumWidth(220)
        self._defect_panel.setMinimumWidth(300)
        self._defect_panel.setMaximumWidth(400)

        splitter.addWidget(self._cam_panel)
        splitter.addWidget(self._video_widget)
        splitter.addWidget(self._defect_panel)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter, stretch=1)

        # ── Controls bar ──────────────────────────────────────────────────
        self._controls = ControlsBar()
        root.addWidget(self._controls)

        # ── Signal wiring ─────────────────────────────────────────────────
        self._cam_panel.connected.connect(self._on_camera_connected)
        self._cam_panel.disconnected.connect(self._on_camera_disconnected)

        self._controls.capture_reference_clicked.connect(self._on_capture_ref)
        self._controls.inspect_toggled.connect(self._on_inspect_toggle)
        self._controls.grid_toggled.connect(self._on_grid_toggle)
        self._controls.diff_toggled.connect(self._on_diff_toggle)
        self._controls.save_snapshot_clicked.connect(self._on_save_snapshot)
        self._controls.generate_report_clicked.connect(self._on_generate_report)

    # ── Camera lifecycle ──────────────────────────────────────────────────────

    @pyqtSlot(int)
    def _on_camera_connected(self, index: int) -> None:
        self._controls.on_camera_connected()
        self._pipeline = InspectionPipeline(self._cfg, self._camera)
        self._pipeline.frame_ready.connect(self._on_frame)
        self._pipeline.defects_ready.connect(self._on_defects)
        self._pipeline.status_update.connect(self._controls.update_status)
        self._pipeline.error.connect(self._on_pipeline_error)
        # Sync initial toggle states
        self._pipeline.set_show_grid(self._controls._btn_grid.isChecked())
        self._pipeline.start()

    @pyqtSlot()
    def _on_camera_disconnected(self) -> None:
        self._controls.on_camera_disconnected()
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline.wait(3000)
            self._pipeline = None
        self._video_widget._show_placeholder()

    # ── Pipeline slots ────────────────────────────────────────────────────────

    @pyqtSlot(object)
    def _on_frame(self, frame: np.ndarray) -> None:
        self._last_frame = frame
        self._video_widget.update_frame(frame)

    @pyqtSlot(object)
    def _on_defects(self, defects: list[Defect]) -> None:
        # Show both YOLO (raw, every frame) and confirmed diff detections.
        # Without a reference, only YOLO runs, so you see live detections immediately.
        self._defect_panel.set_current(defects)

    @pyqtSlot(str)
    def _on_pipeline_error(self, msg: str) -> None:
        self.statusBar().showMessage(f"Error: {msg}", 4000)

    # ── Control actions ───────────────────────────────────────────────────────

    @pyqtSlot()
    def _on_capture_ref(self) -> None:
        if self._pipeline:
            self._pipeline.capture_reference()

    @pyqtSlot(bool)
    def _on_inspect_toggle(self, checked: bool) -> None:
        if self._pipeline:
            self._pipeline.set_inspecting(checked)
        if not checked:
            self._defect_panel.clear()

    @pyqtSlot(bool)
    def _on_grid_toggle(self, checked: bool) -> None:
        if self._pipeline:
            self._pipeline.set_show_grid(checked)

    @pyqtSlot(bool)
    def _on_diff_toggle(self, checked: bool) -> None:
        if self._pipeline:
            self._pipeline.set_show_diff(checked)

    @pyqtSlot()
    def _on_save_snapshot(self) -> None:
        if self._last_frame is None:
            return
        Path(self._cfg.SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = Path(self._cfg.SNAPSHOT_DIR) / f"snapshot_{ts}.png"
        cv2.imwrite(str(fname), self._last_frame)
        self.statusBar().showMessage(f"Snapshot saved: {fname}", 3000)

    @pyqtSlot()
    def _on_generate_report(self) -> None:
        if self._pipeline is None:
            return
        log_path    = self._pipeline.last_log_path
        frame_shape = (self._cfg.OUTPUT_H, self._cfg.OUTPUT_W, 3)
        # Pass the live reference frame so the report can overlay defects on
        # the actual PCB photograph. reference.frame is None if not yet captured.
        ref_img = self._pipeline.reference.frame
        dlg = ReportDialog(log_path, frame_shape,
                           reference_img=ref_img, parent=self)
        dlg.exec_()

    # ── Window close ──────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline.wait(3000)
        self._camera.disconnect()
        event.accept()
