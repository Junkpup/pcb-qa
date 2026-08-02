"""
ui/dialogs/report_dialog.py — Report generation dialog.

Previews the defect concentration map (drawn on the reference PCB image)
and exports the full PDF report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox,
)

from reports.generator import generate_report


class ReportDialog(QDialog):
    """Modal dialog for generating and exporting the inspection report."""

    def __init__(
        self,
        log_path: str,
        frame_shape: tuple,
        reference_img: Optional[np.ndarray] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Generate Report")
        self.setMinimumSize(800, 560)
        self._log_path     = log_path
        self._frame_shape  = frame_shape
        self._reference_img = reference_img      # BGR numpy array or None
        self._preview_path: Path | None = None
        self._build_ui()
        self._render_preview()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel("Inspection Report Preview")
        title.setStyleSheet("font-size:16px; font-weight:bold; color:#e0e0e0;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Sub-label: tells the user what reference is being used
        if self._reference_img is not None:
            hint = "Concentration map overlaid on the captured reference PCB image"
            hint_color = "#66bb6a"
        else:
            hint = "No reference image — showing grid heatmap"
            hint_color = "#ffa726"
        sub = QLabel(hint)
        sub.setStyleSheet(f"font-size:11px; color:{hint_color};")
        sub.setAlignment(Qt.AlignCenter)
        layout.addWidget(sub)

        self._lbl_preview = QLabel("Generating preview…")
        self._lbl_preview.setAlignment(Qt.AlignCenter)
        self._lbl_preview.setMinimumHeight(380)
        self._lbl_preview.setStyleSheet("background:#1a1a1a; border:1px solid #333;")
        layout.addWidget(self._lbl_preview)

        btn_row = QHBoxLayout()
        self._btn_export = QPushButton("Export PDF")
        self._btn_export.setStyleSheet(
            "background:#4a148c; color:white; border-radius:4px;"
            " padding:7px 18px; font-weight:bold;")
        self._btn_export.clicked.connect(self._export_pdf)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_export)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    # ── Preview ───────────────────────────────────────────────────────────────

    def _render_preview(self) -> None:
        try:
            preview_path = Path(self._log_path).parent / "report_preview.png"
            generate_report(
                self._log_path,
                str(preview_path),
                fmt="png",
                frame_shape=self._frame_shape,
                reference_img=self._reference_img,
            )
            self._preview_path = preview_path
            pix = QPixmap(str(preview_path)).scaled(
                760, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._lbl_preview.setPixmap(pix)
        except Exception as exc:
            self._lbl_preview.setText(f"Preview failed: {exc}")

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_pdf(self) -> None:
        default = str(Path(self._log_path).parent / "report.pdf")
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", default, "PDF Files (*.pdf)")
        if not out_path:
            return
        try:
            generate_report(
                self._log_path,
                out_path,
                fmt="pdf",
                frame_shape=self._frame_shape,
                reference_img=self._reference_img,
            )
            QMessageBox.information(
                self, "Exported", f"Report saved to:\n{out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
