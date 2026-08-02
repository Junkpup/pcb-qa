
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

PT_PATH  = Path("models/best.pt")
OUT_DIR  = Path("models")

def main() -> None:
    if not PT_PATH.exists():
        print(f"[export] ✗  {PT_PATH} not found. Run from the project root.")
        sys.exit(1)

    print(f"[export] Loading {PT_PATH} ...")
    model = YOLO(str(PT_PATH))

    print("[export] Exporting to CoreML (this takes ~1–2 minutes) ...")
    # nms=True embeds NMS into the CoreML graph — simpler output format
    export_path = model.export(format="coreml", nms=True, imgsz=640)

    # ultralytics writes the file next to best.pt — move it into models/ cleanly
    exported = Path(export_path)
    target   = OUT_DIR / exported.name
    if exported != target:
        exported.rename(target)
        print(f"[export] ✓  Moved to {target}")
    else:
        print(f"[export] ✓  Exported to {exported}")

    print("\n[export] Done. Restart main.py — it will use CoreML automatically.")


if __name__ == "__main__":
    main()
