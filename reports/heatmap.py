"""
reports/heatmap.py — Defect concentration visualisations.

build_pcb_concentration_map  — PRIMARY: overlays a Gaussian heat field and
    per-severity markers directly on the reference PCB photograph so you can
    see exactly where on the board defects are clustering.

build_heatmap                — FALLBACK: plain grid heatmap used when no
    reference image is available.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter


# ── Severity palette (matches the rest of the UI) ────────────────────────────

_SEV_COLOR = {
    "HIGH":   "#ef5350",
    "MEDIUM": "#ffa726",
    "LOW":    "#66bb6a",
}
_SEV_WEIGHT = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


# ── Primary: PCB photo + heat overlay ────────────────────────────────────────

def build_pcb_concentration_map(
    defect_rows: list[dict],
    reference_img: np.ndarray,          # BGR numpy array from OpenCV
    frame_shape: tuple,                 # (height, width, channels)
    sigma: int = 35,                    # Gaussian spread in pixels
) -> matplotlib.figure.Figure:
    """
    Render a defect concentration map overlaid on the reference PCB image.

    Parameters
    ----------
    defect_rows   : list of dicts — keys: x, y, w, h, label, severity
    reference_img : BGR numpy array captured as the inspection reference
    frame_shape   : (height, width, channels) of the inspection feed
    sigma         : Gaussian blur radius controlling heat spread (px)

    Returns
    -------
    matplotlib Figure — caller is responsible for closing it.
    """
    h, w = frame_shape[:2]

    # ── Prepare background ────────────────────────────────────────────────
    ref_resized = cv2.resize(reference_img, (w, h))        # ensure correct size
    ref_rgb     = ref_resized[:, :, ::-1].copy()            # BGR → RGB for matplotlib

    # ── Build weighted heat accumulation grid ─────────────────────────────
    heat = np.zeros((h, w), dtype=np.float32)
    for d in defect_rows:
        cx = int(d["x"]) + int(d["w"]) // 2
        cy = int(d["y"]) + int(d["h"]) // 2
        cx = int(np.clip(cx, 0, w - 1))
        cy = int(np.clip(cy, 0, h - 1))
        weight = _SEV_WEIGHT.get(str(d.get("severity", "LOW")), 1)
        heat[cy, cx] += weight

    has_defects = heat.max() > 0
    if has_defects:
        heat = gaussian_filter(heat, sigma=sigma)
        heat = heat / heat.max()          # normalise to [0, 1]

    # ── Compose the figure ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    # Layer 1: reference PCB photograph
    ax.imshow(ref_rgb)

    # Layer 2: semi-transparent heat overlay (only when defects exist)
    if has_defects:
        cmap      = plt.get_cmap("YlOrRd")
        heat_rgba = cmap(heat)                  # (h, w, 4) RGBA float
        # Alpha modulated by local intensity — zero heat = fully transparent
        heat_rgba[..., 3] = heat * 0.60
        ax.imshow(heat_rgba, extent=[0, w, h, 0])

        # Colour bar
        sm = plt.cm.ScalarMappable(
            cmap="YlOrRd", norm=Normalize(vmin=0, vmax=heat.max()))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.01)
        cbar.set_label("Relative Defect Concentration", color="#bbbbbb", fontsize=10)
        cbar.ax.yaxis.set_tick_params(color="#aaaaaa")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaaaaa")

    # Layer 3: per-defect markers (circles), colour-coded by severity
    plotted_labels: set[str] = set()
    for d in defect_rows:
        cx   = int(d["x"]) + int(d["w"]) // 2
        cy   = int(d["y"]) + int(d["h"]) // 2
        sev  = str(d.get("severity", "LOW"))
        col  = _SEV_COLOR.get(sev, "#ffffff")
        lbl  = str(d.get("label", "unknown"))
        ax.plot(cx, cy, "o",
                color=col, markersize=9, alpha=0.85,
                markeredgecolor="white", markeredgewidth=0.8,
                zorder=5)
        plotted_labels.add(sev)

    # Legend for severity colours
    legend_handles = [
        mpatches.Patch(facecolor=_SEV_COLOR[s], edgecolor="white",
                       linewidth=0.6, label=s)
        for s in ("HIGH", "MEDIUM", "LOW")
        if s in plotted_labels
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right",
                  framealpha=0.6, facecolor="#1e1e1e",
                  edgecolor="#555", labelcolor="white", fontsize=10)

    if not has_defects:
        ax.text(0.5, 0.5, "No defects recorded",
                ha="center", va="center", transform=ax.transAxes,
                color="#888888", fontsize=14,
                bbox=dict(boxstyle="round", facecolor="#1e1e1e",
                          alpha=0.7, edgecolor="#444"))

    n = len(defect_rows)
    ax.set_title(
        f"Defect Concentration Map — {n} detection{'s' if n != 1 else ''} "
        f"on Reference PCB",
        color="white", fontsize=13, pad=10,
    )
    ax.axis("off")
    plt.tight_layout()
    return fig


# ── Fallback: plain grid heatmap (no reference image) ────────────────────────

def build_heatmap(
    defect_rows: list[dict],
    frame_shape: tuple,
    grid_px: int = 64,
) -> matplotlib.figure.Figure:
    """
    Grid-based heatmap used when no reference photograph is available.

    Parameters
    ----------
    defect_rows  : list of dicts with keys x, y, w, h
    frame_shape  : (height, width, channels)
    grid_px      : pixel size of each heatmap cell
    """
    import seaborn as sns

    h, w = frame_shape[:2]
    rows = max(1, h // grid_px)
    cols = max(1, w // grid_px)
    grid = np.zeros((rows, cols), dtype=float)

    for d in defect_rows:
        cx = int(d["x"]) + int(d["w"]) // 2
        cy = int(d["y"]) + int(d["h"]) // 2
        r  = min(cy // grid_px, rows - 1)
        c  = min(cx // grid_px, cols - 1)
        grid[r, c] += 1

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    sns.heatmap(
        grid, ax=ax,
        cmap="YlOrRd", linewidths=0.3, linecolor="#333333",
        annot=True, fmt=".0f",
        cbar_kws={"label": "Defect Count"},
    )
    ax.set_title("Defect Concentration Map (no reference image)",
                 color="white", fontsize=14, pad=12)
    ax.set_xlabel(f"X grid cells ({grid_px} px each)", color="#aaa")
    ax.set_ylabel(f"Y grid cells ({grid_px} px each)", color="#aaa")
    ax.tick_params(colors="#aaa")
    plt.tight_layout()
    return fig
