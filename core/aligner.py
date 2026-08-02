"""
core/aligner.py — ORB feature matching and homography-based alignment.

Migrated from aoi_system.py:FeatureMatcher and HomographyAligner.

Homography H maps the live frame onto the reference plane:
  x_ref = H · x_live   (homogeneous coordinates)

H = [[h11, h12, h13],
     [h21, h22, h23],
     [h31, h32, h33]]

Estimated with RANSAC to tolerate outlier correspondences.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from config import Cfg
from core.reference import ReferenceManager


class FeatureMatcher:
    """Matches ORB keypoints between the reference and a live frame."""

    def __init__(self, cfg: Cfg):
        self._cfg = cfg
        self._orb = cv2.ORB_create(nfeatures=cfg.ORB_FEATURES)
        self._bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(
        self,
        ref: ReferenceManager,
        live_gray: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Returns (src_pts, dst_pts, n_good_matches).
        src_pts — keypoint coordinates in the reference frame
        dst_pts — corresponding coordinates in the live frame
        Returns (None, None, 0) if insufficient matches are found.
        """
        kp2, desc2 = self._orb.detectAndCompute(live_gray, None)

        if desc2 is None or ref.desc is None or len(kp2) < 2:
            return None, None, 0

        try:
            raw = self._bf.knnMatch(ref.desc, desc2, k=2)
        except cv2.error:
            return None, None, 0

        good = [m for pair in raw if len(pair) == 2
                for m, n in [pair]
                if m.distance < self._cfg.LOWE_RATIO * n.distance]

        if len(good) < 4:
            return None, None, len(good)

        src = np.float32([ref.kp[m.queryIdx].pt for m in good])
        dst = np.float32([kp2[m.trainIdx].pt    for m in good])
        return src, dst, len(good)


class HomographyAligner:
    """Warps the live frame onto the reference plane via RANSAC homography."""

    def __init__(self, cfg: Cfg):
        self._cfg = cfg

    def align(
        self,
        live_frame: np.ndarray,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        ref_shape: tuple,
    ) -> tuple[Optional[np.ndarray], float]:
        """
        Returns (warped_frame, reprojection_error_px) or (None, inf) on failure.
        """
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        if H is None:
            return None, float("inf")

        det = abs(np.linalg.det(H))
        if not (self._cfg.H_DET_MIN < det < self._cfg.H_DET_MAX):
            return None, float("inf")

        inliers = mask.ravel() == 1
        if inliers.sum() < 4:
            return None, float("inf")

        src_in = src_pts[inliers]
        dst_in = dst_pts[inliers]
        proj   = cv2.perspectiveTransform(dst_in.reshape(-1, 1, 2), H).reshape(-1, 2)
        err    = float(np.sqrt(np.mean(np.sum((src_in - proj) ** 2, axis=1))))

        if err > self._cfg.H_REPROJ_MAX:
            return None, err

        h, w = ref_shape[:2]
        warped = cv2.warpPerspective(live_frame, H, (w, h))
        return warped, err
