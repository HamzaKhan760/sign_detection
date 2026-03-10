"""
Export trained YOLOv8 to TFLite (int8) for Coral Edge TPU.

Usage: python scripts/export_yolo.py

After this, compile for Edge TPU on Linux/Pi/Colab:
    edgetpu_compiler exported_model/best_full_integer_quant.tflite
"""

from ultralytics import YOLO
import shutil
import os

# ─── CONFIG ──────────────────────────────────────────────────────────────────

BEST_MODEL = "runs/detect/runs/sign_detection/weights/best.pt"
EXPORT_DIR = "exported_model"
IMG_SIZE = 320

# ─── EXPORT ──────────────────────────────────────────────────────────────────

os.makedirs(EXPORT_DIR, exist_ok=True)

model = YOLO(BEST_MODEL)

# Export to TFLite with int8 quantization
# YOLOv8 handles the full conversion pipeline internally
print("Exporting to TFLite with int8 quantization...")
print("This may take a few minutes...\n")

model.export(
    format="tflite",
    imgsz=IMG_SIZE,
    int8=True,              # full integer quantization for Coral
    data="dataset/dataset.yaml",  # needed for calibration during int8 quantization
)

# Find the exported file and copy to export directory
# YOLOv8 saves it next to the .pt file
export_search_dir = os.path.dirname(BEST_MODEL)
for root, dirs, files in os.walk(export_search_dir):
    for f in files:
        if f.endswith("_int8.tflite") or f.endswith("_full_integer_quant.tflite"):
            src = os.path.join(root, f)
            dst = os.path.join(EXPORT_DIR, f)
            shutil.copy2(src, dst)
            print(f"\nCopied: {src} → {dst}")

# Also copy the regular tflite if it exists
for root, dirs, files in os.walk(export_search_dir):
    for f in files:
        if f.endswith(".tflite") and "int8" not in f and "integer" not in f:
            src = os.path.join(root, f)
            dst = os.path.join(EXPORT_DIR, f)
            shutil.copy2(src, dst)

print(f"\nExport complete! Files in: {EXPORT_DIR}/")
print(f"""
{'='*60}
NEXT: Compile for Edge TPU
{'='*60}

Transfer the int8 .tflite file to a Linux machine or Raspberry Pi, then:

  edgetpu_compiler {EXPORT_DIR}/best_full_integer_quant.tflite

Or use Google Colab:
  !pip install edgetpu-compiler
  !edgetpu_compiler best_full_integer_quant.tflite

This produces *_edgetpu.tflite — that's what runs on the Coral.
""")
