"""
core/camera.py — Camera source abstraction.

Works with any device that cv2.VideoCapture accepts: Camo virtual webcam,
USB capture card (Nikon D300 via HDMI), or built-in camera.
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np

from config import Cfg


class CameraSource:
    """Thread-safe wrapper around cv2.VideoCapture."""

    def __init__(self, cfg: Cfg):
        self._cfg = cfg
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()

    # ── Device discovery ──────────────────────────────────────────────────────

    @staticmethod
    def list_devices(max_index: int = 9) -> list[tuple[int, str]]:
        """
        Scan camera indices 0–max_index and return (index, label) pairs
        for each available device.
        """
        devices: list[tuple[int, str]] = []
        for i in range(max_index + 1):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    devices.append((i, f"Camera {i}"))
                cap.release()
        return devices

    # ── Connection lifecycle ──────────────────────────────────────────────────

    def connect(self, index: int) -> bool:
        """Open the camera at the given index. Returns True on success."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()

            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                return False

            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._cfg.OUTPUT_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.OUTPUT_H)
            cap.set(cv2.CAP_PROP_FPS,          self._cfg.TARGET_FPS)
            self._cap = cap
            self._cfg.CAMERA_INDEX = index
            return True

    def disconnect(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    @property
    def is_connected(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ── Frame access ──────────────────────────────────────────────────────────

    def read(self) -> Optional[np.ndarray]:
        """Return the latest BGR frame, or None if not available."""
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return None
            ret, frame = self._cap.read()
            return frame if ret else None

    @property
    def resolution(self) -> tuple[int, int]:
        """Return (width, height) of the connected camera."""
        if self._cap is None:
            return (0, 0)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 0.0
        return float(self._cap.get(cv2.CAP_PROP_FPS))
