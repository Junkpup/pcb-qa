"""
core/logger.py — Session-scoped CSV defect logger.
Migrated and extended from aoi_system.py:CSVLogger.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from config import Cfg
from core.detector import Defect

_FIELDS = ["timestamp", "frame", "label", "severity",
           "x", "y", "w", "h", "area", "confidence", "source"]


class CSVLogger:
    """Appends defect rows to a per-session CSV file under data/logs/."""

    def __init__(self, cfg: Cfg):
        self._cfg = cfg
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = Path(cfg.LOG_DIR) / f"session_{ts}.csv"
        self._write_header()
        print(f"[logger] ✓ Logging to {self.path}")

    def _write_header(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(_FIELDS)

    def log(self, defects: list[Defect], frame_n: int) -> None:
        if not defects:
            return
        ts = datetime.now().isoformat(timespec="seconds")
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            for d in defects:
                w.writerow(d.as_row(ts, frame_n))
