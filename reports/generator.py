"""
reports/generator.py — Session report generator.

Reads the session CSV log and produces:
  • Defect concentration map — overlaid on the reference PCB image
    (falls back to a plain grid heatmap when no reference is provided)
  • Bar chart: defect counts by type
  • Pie chart: severity distribution
  • Summary text with overall PASS / WARN / FAIL verdict

Output formats: 'png' (preview) or 'pdf' (export).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe from any thread
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from reports.heatmap import build_pcb_concentration_map, build_heatmap

_SEVERITY_COLORS = {
    "HIGH":   "#ef5350",
    "MEDIUM": "#ffa726",
    "LOW":    "#66bb6a",
}


def generate_report(
    log_path: str,
    output_path: str,
    fmt: str = "pdf",
    frame_shape: tuple = (720, 1280, 3),
    reference_img: Optional[np.ndarray] = None,
) -> None:
    """
    Parameters
    ----------
    log_path      : path to the session CSV defect log
    output_path   : destination file (.pdf or .png)
    fmt           : 'pdf' or 'png'
    frame_shape   : (height, width, channels) of the inspection feed
    reference_img : BGR numpy array of the captured reference PCB frame.
                    When provided, the concentration map is drawn on top of
                    the actual PCB photograph.
    """
    df = pd.read_csv(log_path)

    if fmt == "pdf":
        _export_pdf(df, output_path, frame_shape, reference_img)
    else:
        _export_png(df, output_path, frame_shape, reference_img)


# ── Figure builders ───────────────────────────────────────────────────────────

def _build_figures(
    df: pd.DataFrame,
    frame_shape: tuple,
    reference_img: Optional[np.ndarray],
) -> list:
    """Build all report figures. Returns list of matplotlib Figure objects."""
    figures = []
    plt.style.use("dark_background")

    # ── Figure 1: Defect concentration map ───────────────────────────────
    defect_rows = df[["x", "y", "w", "h", "label", "severity"]].to_dict("records") \
        if not df.empty else []

    if reference_img is not None:
        # Primary path: heat overlay on the real PCB photograph
        fig_hm = build_pcb_concentration_map(defect_rows, reference_img, frame_shape)
    else:
        # Fallback: plain grid heatmap
        if defect_rows:
            fig_hm = build_heatmap(defect_rows, frame_shape)
        else:
            fig_hm, ax = plt.subplots(figsize=(10, 6))
            fig_hm.patch.set_facecolor("#1e1e1e")
            ax.text(0.5, 0.5, "No defects recorded",
                    ha="center", va="center",
                    transform=ax.transAxes, color="#aaa", fontsize=14)
            ax.set_title("Defect Concentration Map", color="white")
    figures.append(fig_hm)

    # ── Figure 2: Defect type bar + severity pie ──────────────────────────
    fig2, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(12, 5))
    fig2.patch.set_facecolor("#1e1e1e")

    if not df.empty:
        counts = df["label"].value_counts()
        colors = [
            _SEVERITY_COLORS.get(
                df[df["label"] == lbl]["severity"].mode().iloc[0], "#9e9e9e")
            for lbl in counts.index
        ]
        ax_bar.barh(counts.index, counts.values, color=colors)
        ax_bar.set_xlabel("Count", color="#aaa")
        ax_bar.set_title("Defects by Type", color="white")
        ax_bar.tick_params(colors="#aaa")
        ax_bar.set_facecolor("#1e1e1e")

        sev_counts = df["severity"].value_counts()
        pie_colors = [_SEVERITY_COLORS.get(s, "#9e9e9e") for s in sev_counts.index]
        ax_pie.pie(
            sev_counts.values,
            labels=sev_counts.index,
            colors=pie_colors,
            autopct="%1.0f%%",
            textprops={"color": "white"},
        )
        ax_pie.set_title("Severity Distribution", color="white")
    else:
        for ax in (ax_bar, ax_pie):
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="#aaa")
        ax_bar.set_title("Defects by Type", color="white")
        ax_pie.set_title("Severity Distribution", color="white")

    plt.tight_layout()
    figures.append(fig2)

    # ── Figure 3: Summary ─────────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    fig3.patch.set_facecolor("#1e1e1e")
    ax3.axis("off")

    total  = len(df)
    high   = int((df["severity"] == "HIGH").sum())   if not df.empty else 0
    medium = int((df["severity"] == "MEDIUM").sum()) if not df.empty else 0
    low    = int((df["severity"] == "LOW").sum())    if not df.empty else 0
    verdict = "FAIL" if high > 0 else ("WARN" if medium > 0 else "PASS")
    v_color = "#ef5350" if verdict == "FAIL" else \
              "#ffa726" if verdict == "WARN" else "#66bb6a"

    summary = (
        f"Session Summary\n"
        f"{'─' * 40}\n"
        f"Total Defects : {total}\n"
        f"  HIGH        : {high}\n"
        f"  MEDIUM      : {medium}\n"
        f"  LOW         : {low}\n"
        f"\nOverall Verdict: {verdict}"
    )
    ax3.text(0.05, 0.95, summary,
             transform=ax3.transAxes,
             verticalalignment="top",
             fontsize=13, color="white", fontfamily="monospace")
    ax3.text(0.75, 0.3, verdict,
             transform=ax3.transAxes,
             fontsize=48, fontweight="bold",
             color=v_color, ha="center")
    figures.append(fig3)

    return figures


# ── Export helpers ────────────────────────────────────────────────────────────

def _export_pdf(
    df: pd.DataFrame,
    path: str,
    frame_shape: tuple,
    reference_img: Optional[np.ndarray],
) -> None:
    figs = _build_figures(df, frame_shape, reference_img)
    with PdfPages(path) as pdf:
        for fig in figs:
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)


def _export_png(
    df: pd.DataFrame,
    path: str,
    frame_shape: tuple,
    reference_img: Optional[np.ndarray],
) -> None:
    import io
    import cv2

    figs = _build_figures(df, frame_shape, reference_img)
    images = []
    for fig in figs:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        arr = np.frombuffer(buf.read(), dtype=np.uint8)
        images.append(cv2.imdecode(arr, cv2.IMREAD_COLOR))

    if not images:
        return
    max_w  = max(im.shape[1] for im in images)
    padded = [
        cv2.copyMakeBorder(im, 0, 0, 0, max_w - im.shape[1],
                           cv2.BORDER_CONSTANT, value=(30, 30, 30))
        for im in images
    ]
    cv2.imwrite(path, np.vstack(padded))
