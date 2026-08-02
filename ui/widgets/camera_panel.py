"""
ui/widgets/camera_panel.py — Camera connection panel.
Scans available cv2.VideoCapture devices and lets the user connect/disconnect.
"""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QGroupBox,
)

from core.camera import CameraSource
from config import Cfg


class CameraPanel(QGroupBox):
    """Left-panel widget for camera selection and connection."""

    connected    = pyqtSignal(int)   # emits camera index when connected
    disconnected = pyqtSignal()

    def __init__(self, cfg: Cfg, camera: CameraSource, parent=None):
        super().__init__("Camera", parent)
        self._cfg    = cfg
        self._camera = camera
        self._build_ui()
        self.refresh_devices()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Device dropdown + refresh
        row = QHBoxLayout()
        self._combo = QComboBox()
        self._combo.setMinimumWidth(160)
        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.setFixedWidth(70)
        self._btn_refresh.clicked.connect(self.refresh_devices)
        row.addWidget(self._combo)
        row.addWidget(self._btn_refresh)
        layout.addLayout(row)

        # Connect / Disconnect
        self._btn_connect = QPushButton("Connect")
        self._btn_connect.setStyleSheet(
            "QPushButton { background:#2e7d32; color:white; border-radius:4px; padding:5px; }"
            "QPushButton:hover { background:#388e3c; }"
            "QPushButton:disabled { background:#444; color:#888; }"
        )
        self._btn_connect.clicked.connect(self._on_connect)
        layout.addWidget(self._btn_connect)

        self._btn_disconnect = QPushButton("Disconnect")
        self._btn_disconnect.setEnabled(False)
        self._btn_disconnect.setStyleSheet(
            "QPushButton { background:#c62828; color:white; border-radius:4px; padding:5px; }"
            "QPushButton:hover { background:#d32f2f; }"
            "QPushButton:disabled { background:#444; color:#888; }"
        )
        self._btn_disconnect.clicked.connect(self._on_disconnect)
        layout.addWidget(self._btn_disconnect)

        # Status info
        self._lbl_status = QLabel("Not connected")
        self._lbl_status.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        layout.addWidget(self._lbl_status)

        layout.addStretch()

    # ── Slots ──────────────────────────────────────────────────────────────────

    def refresh_devices(self) -> None:
        self._combo.clear()
        devices = CameraSource.list_devices()
        for idx, label in devices:
            self._combo.addItem(f"{label}  (index {idx})", userData=idx)
        if not devices:
            self._combo.addItem("No cameras found", userData=None)
            self._btn_connect.setEnabled(False)
        else:
            self._btn_connect.setEnabled(True)

    def _on_connect(self) -> None:
        idx = self._combo.currentData()
        if idx is None:
            return
        ok = self._camera.connect(idx)
        if ok:
            w, h = self._camera.resolution
            fps  = self._camera.fps
            self._lbl_status.setText(f"Connected: {w}×{h} @ {fps:.0f} fps")
            self._lbl_status.setStyleSheet("color: #66bb6a; font-size: 11px;")
            self._btn_connect.setEnabled(False)
            self._btn_disconnect.setEnabled(True)
            self.connected.emit(idx)
        else:
            self._lbl_status.setText("Failed to open camera")
            self._lbl_status.setStyleSheet("color: #ef5350; font-size: 11px;")

    def _on_disconnect(self) -> None:
        self._camera.disconnect()
        self._lbl_status.setText("Disconnected")
        self._lbl_status.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        self._btn_connect.setEnabled(True)
        self._btn_disconnect.setEnabled(False)
        self.disconnected.emit()
