"""
main.py - Entry point for the AOI PCB Inspection System.

Usage
-----
  python main.py                         # Launch the GUI (standard mode)
  python main.py --camera-index 1        # Pre-select camera index
  python main.py --ref data/references/ref.png   # Pre-load a reference image

First run — install dependencies:
  pip install -r requirements.txt

Camera setup:
  • Camo (phone as webcam): open Camo on your phone, enable USB connection,
    run `python main.py` and select the Camo device from the dropdown.
  • Nikon D300 via HDMI capture card: connect the card, select it from the dropdown.
  • Built-in camera or any USB webcam: appears automatically in the dropdown.

Quick start:
  1. Select your camera in the left panel and click Connect.
  2. Point camera at a known-good PCB, click Capture Reference.
  3. Click Start Inspection — defects appear in real time.
  4. Click Generate Report when done.
"""

import sys
import argparse


import torch                          
from ultralytics import YOLO          

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from config import Cfg
from ui.main_window import MainWindow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AOI PCB Inspection System")
    p.add_argument("--camera-index", type=int, default=None,
                   help="Pre-select camera index on startup")
    p.add_argument("--ref", default=None,
                   help="Path to a reference PCB image to pre-load")
    p.add_argument("--width",  type=int, default=None, help="Output frame width")
    p.add_argument("--height", type=int, default=None, help="Output frame height")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Cfg()
    if args.width:
        cfg.OUTPUT_W = args.width
    if args.height:
        cfg.OUTPUT_H = args.height
    if args.camera_index is not None:
        cfg.CAMERA_INDEX = args.camera_index

    cfg.ensure_dirs()

    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setApplicationName("AOI PCB Inspection")
    app.setStyle("Fusion")

    # Dark palette
    from PyQt5.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(30,  30,  30))
    palette.setColor(QPalette.WindowText,      QColor(224, 224, 224))
    palette.setColor(QPalette.Base,            QColor(18,  18,  18))
    palette.setColor(QPalette.AlternateBase,   QColor(35,  35,  35))
    palette.setColor(QPalette.ToolTipBase,     QColor(50,  50,  50))
    palette.setColor(QPalette.ToolTipText,     QColor(224, 224, 224))
    palette.setColor(QPalette.Text,            QColor(224, 224, 224))
    palette.setColor(QPalette.Button,          QColor(45,  45,  45))
    palette.setColor(QPalette.ButtonText,      QColor(224, 224, 224))
    palette.setColor(QPalette.Highlight,       QColor(42,  130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0,   0,   0))
    app.setPalette(palette)

    window = MainWindow(cfg)

    # Pre-load reference if supplied via CLI
    if args.ref and window._pipeline is None:
        # Will be applied when the pipeline starts after camera connection
        window._pending_cli_ref = args.ref
    else:
        window._pending_cli_ref = None

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
