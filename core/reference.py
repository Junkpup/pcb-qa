"""
core/reference.py — Stores and manages the known-good reference PCB frame.
Migrated from aoi_system.py:ReferenceManager.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import Cfg


class ReferenceManager:
    """Holds the reference PCB frame and its pre-computed ORB descriptors."""

    def __init__(self, cfg: Cfg):
        self._cfg = cfg
        self._orb = cv2.ORB_create(nfeatures=cfg.ORB_FEATURES)
        self.reset()

    def reset(self) -> None:
        self.frame: Optional[np.ndarray] = None
        self.gray:  Optional[np.ndarray] = None
        self.kp:    list = []
        self.desc:  Optional[np.ndarray] = None

    @property
    def ready(self) -> bool:
        return self.frame is not None and self.desc is not None

    def set(self, frame: np.ndarray) -> "ReferenceManager":
        """Capture frame as the new reference and compute ORB features."""
        self.frame = frame.copy()
        self.gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.kp, self.desc = self._orb.detectAndCompute(self.gray, None)
        n = len(self.kp)
        print(f"[reference] ✓ Captured — {n} ORB keypoints.")
        if n < self._cfg.ORB_MIN_MATCHES:
            print(f"[reference] ⚠  Only {n} keypoints. "
                  "Improve lighting or PCB texture for better matching.")
        return self

    def load(self, path: str | Path) -> "ReferenceManager":
        """Load a saved reference image from disk."""
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot read reference image: {path}")
        img = cv2.resize(img, (self._cfg.OUTPUT_W, self._cfg.OUTPUT_H))
        self.set(img)
        print(f"[reference] ✓ Loaded from {path}")
        return self

    def save(self, path: str | Path) -> None:
        """Persist the current reference frame to disk."""
        if self.frame is None:
            raise RuntimeError("No reference to save.")
        cv2.imwrite(str(path), self.frame)
        print(f"[reference] ✓ Saved to {path}")
