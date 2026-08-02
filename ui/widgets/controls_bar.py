"""
ui/widgets/controls_bar.py — Toolbar + status bar at the bottom of the window.
"""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFrame,
)


def _btn(text: str, color: str, checkable: bool = False) -> QPushButton:
    b = QPushButton(text)
    b.setCheckable(checkable)
    b.setStyleSheet(
        f"QPushButton {{ background:{color}; color:white; border-radius:4px; "
        f"padding:6px 12px; font-weight:bold; }}"
        f"QPushButton:hover {{ filter:brightness(120%); }}"
        f"QPushButton:checked {{ border:2px solid white; }}"
        f"QPushButton:disabled {{ background:#444; color:#888; }}"
    )
    return b


class ControlsBar(QWidget):
    """
    Toolbar with inspection controls and a live status strip.

    Signals
    -------
    capture_reference_clicked
    inspect_toggled(bool)
    grid_toggled(bool)
    diff_toggled(bool)
    save_snapshot_clicked
    generate_report_clicked
    """

    capture_reference_clicked = pyqtSignal()
    inspect_toggled            = pyqtSignal(bool)
    grid_toggled               = pyqtSignal(bool)
    diff_toggled               = pyqtSignal(bool)
    save_snapshot_clicked      = pyqtSignal()
    generate_report_clicked    = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:#212121;")
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 4, 8, 4)
        outer.setSpacing(4)

        # ── Toolbar buttons ────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._btn_ref     = _btn("Capture Reference (optional)", "#1565c0")
        self._btn_inspect = _btn("Start Inspection",  "#6a1aae", checkable=True)
        self._btn_grid    = _btn("Grid",              "#37474f", checkable=True)
        self._btn_diff    = _btn("Diff Map",          "#37474f", checkable=True)
        self._btn_save    = _btn("Save Snapshot",     "#33691e")
        self._btn_report  = _btn("Generate Report",   "#4a148c")

        self._btn_ref.setEnabled(False)
        self._btn_inspect.setEnabled(False)    # enabled once camera connects
        self._btn_save.setEnabled(False)
        self._btn_report.setEnabled(False)
        self._btn_grid.setChecked(True)

        self._btn_ref.clicked.connect(self.capture_reference_clicked)
        self._btn_inspect.toggled.connect(self._on_inspect_toggle)
        self._btn_grid.toggled.connect(self.grid_toggled)
        self._btn_diff.toggled.connect(self.diff_toggled)
        self._btn_save.clicked.connect(self.save_snapshot_clicked)
        self._btn_report.clicked.connect(self.generate_report_clicked)

        for b in (self._btn_ref, self._btn_inspect, self._btn_grid,
                  self._btn_diff, self._btn_save, self._btn_report):
            btn_row.addWidget(b)
        btn_row.addStretch()
        outer.addLayout(btn_row)

        # ── Status strip ───────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#333;")
        outer.addWidget(sep)

        status_row = QHBoxLayout()
        status_row.setSpacing(20)

        def _stat(label: str) -> QLabel:
            lbl = QLabel(label)
            lbl.setStyleSheet("color:#9e9e9e; font-size:11px;")
            return lbl

        self._lbl_fps     = _stat("FPS: —")
        self._lbl_matches = _stat("Matches: —")
        self._lbl_reproj  = _stat("Reproj: —")
        self._lbl_verdict = _stat("Verdict: —")
        self._lbl_verdict.setStyleSheet("font-size:12px; font-weight:bold; color:#aaa;")

        for lbl in (self._lbl_fps, self._lbl_matches,
                    self._lbl_reproj, self._lbl_verdict):
            status_row.addWidget(lbl)
        status_row.addStretch()
        outer.addLayout(status_row)

    # ── Public API ────────────────────────────────────────────────────────────

    def on_camera_connected(self) -> None:
        self._btn_ref.setEnabled(True)
        self._btn_inspect.setEnabled(True)
        self._btn_save.setEnabled(True)
        self._btn_report.setEnabled(True)

    def on_camera_disconnected(self) -> None:
        self._btn_ref.setEnabled(False)
        self._btn_inspect.setEnabled(False)
        self._btn_save.setEnabled(False)

    def update_status(self, state: dict) -> None:
        self._lbl_fps.setText(f"FPS: {state.get('fps', 0):.1f}")
        if state.get("ref_ready"):
            self._lbl_matches.setText(f"Matches: {state.get('match_count', 0)}")
            self._lbl_reproj.setText(f"Reproj: {state.get('reproj_err', 0):.1f} px")
        verdict = state.get("verdict", "")
        color = "#66bb6a" if verdict == "PASS" else \
                "#ef5350" if verdict == "FAIL" else "#aaa"
        self._lbl_verdict.setText(f"Verdict: {verdict or '—'}")
        self._lbl_verdict.setStyleSheet(
            f"font-size:12px; font-weight:bold; color:{color};")

    # ── Internal slots ────────────────────────────────────────────────────────

    def _on_inspect_toggle(self, checked: bool) -> None:
        self._btn_inspect.setText("Stop Inspection" if checked else "Start Inspection")
        self.inspect_toggled.emit(checked)
