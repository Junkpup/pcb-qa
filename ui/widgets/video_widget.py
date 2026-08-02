"""
ui/widgets/video_widget.py — Live video display widget.
Receives BGR np.ndarray frames and renders them in a QLabel,
preserving aspect ratio.
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy


class VideoWidget(QLabel):
    """Displays OpenCV BGR frames via Qt's QLabel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #1a1a1a;")
        self.setMinimumSize(640, 360)
        self._show_placeholder()

    def update_frame(self, frame: np.ndarray) -> None:
        """Convert a BGR frame to QPixmap and display it."""
        rgb = frame[:, :, ::-1].copy()           # BGR → RGB
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,   # SmoothTransformation was bottlenecking on older GPUs
        )
        self.setPixmap(pix)

    def _show_placeholder(self) -> None:
        self.setText("No camera connected\n\nSelect a device and click Connect")
        self.setStyleSheet(
            "background-color: #1a1a1a; color: #555555; font-size: 14px;"
        )
