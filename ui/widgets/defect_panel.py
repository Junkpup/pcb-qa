"""
ui/widgets/defect_panel.py — Live defect list sidebar.
Displays every detected defect in a colour-coded table.
"""

from __future__ import annotations

from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTableWidget, QTableWidgetItem,
    QGroupBox, QLabel,
)

from core.detector import Defect


_SEVERITY_BG = {
    "HIGH":   QColor(180, 30,  30,  180),
    "MEDIUM": QColor(180, 100, 0,   180),
    "LOW":    QColor(0,   120, 100, 180),
}

_COLUMNS = ["Time", "Label", "Severity", "Conf", "Source", "Area (px²)"]


class DefectPanel(QGroupBox):
    """Right-panel widget: colour-coded defect log table."""

    def __init__(self, parent=None):
        super().__init__("Defects", parent)
        self._total    = 0   # cumulative confirmed defect count this session
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Summary label
        self._lbl_summary = QLabel("No defects detected")
        self._lbl_summary.setStyleSheet("color: #66bb6a; font-size: 12px; font-weight: bold;")
        layout.addWidget(self._lbl_summary)

        # Table
        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            "QTableWidget { background:#1e1e1e; color:#e0e0e0; gridline-color:#333; }"
            "QHeaderView::section { background:#2a2a2a; color:#ccc; border:1px solid #333; }"
            "QTableWidget::item:alternate { background:#252525; }"
        )
        self._table.setColumnWidth(0, 70)   # Time
        self._table.setColumnWidth(1, 130)  # Label
        self._table.setColumnWidth(2, 65)   # Severity
        self._table.setColumnWidth(3, 45)   # Conf
        self._table.setColumnWidth(4, 45)   # Source
        layout.addWidget(self._table)

        # Buttons
        btn_row = QHBoxLayout()
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self.clear)
        btn_row.addWidget(btn_clear)
        layout.addLayout(btn_row)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_current(self, defects: list[Defect]) -> None:
        """Replace the table with the current frame's confirmed defects.

        The table always shows only the live set — no historical accumulation.
        _total tracks how many confirmed defects have been reported this session.
        """
        self._table.setRowCount(0)
        if not defects:
            self._update_summary()
            return

        ts = datetime.now().strftime("%H:%M:%S")
        for d in defects:
            row = self._table.rowCount()
            self._table.insertRow(row)
            bg = _SEVERITY_BG.get(d.severity, QColor(60, 60, 60, 180))
            values = [ts, d.label, d.severity, f"{d.confidence:.0%}",
                      d.source, str(int(d.area))]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                item.setBackground(bg)
                self._table.setItem(row, col, item)

        self._total += len(defects)
        self._update_summary()

    def clear(self) -> None:
        self._table.setRowCount(0)
        self._total = 0
        self._update_summary()

    def _update_summary(self) -> None:
        if self._total == 0:
            self._lbl_summary.setText("No defects detected")
            self._lbl_summary.setStyleSheet(
                "color: #66bb6a; font-size: 12px; font-weight: bold;")
        else:
            self._lbl_summary.setText(f"{self._total} defect(s) this session")
            self._lbl_summary.setStyleSheet(
                "color: #ef5350; font-size: 12px; font-weight: bold;")
