"""
STEP 8: Export to TFLite (int8 quantized) for Coral Edge TPU.

Usage: python scripts/export_tflite.py

This script:
1. Exports the trained checkpoint to a SavedModel
2. Converts to TFLite with full integer (int8) quantization
3. Prints instructions for Edge TPU compilation
"""

import os
import subprocess
import sys
import numpy as np
import tensorflow as tf


# ─── CONFIG ──────────────────────────────────────────────────────────────────

PIPELINE_CONFIG = "training/pipeline.config"
CHECKPOINT_DIR = "training/train_output"
EXPORT_DIR = "exported_model"
TFLITE_OUTPUT = "exported_model/model.tflite"

# Representative dataset images for quantization calibration
TRAIN_TFRECORD = "training/train/train.tfrecord"
NUM_CALIBRATION_IMAGES = 100
INPUT_SIZE = 320

# ─── STEP 1: EXPORT SAVED MODEL ─────────────────────────────────────────────

print("Step 1: Exporting SavedModel...")

# Find exporter script
exporter_candidates = [
    "models/research/object_detection/exporter_main_v2.py",
]

try:
    import object_detection
    pkg_dir = os.path.dirname(object_detection.__file__)
    exporter_candidates.append(os.path.join(pkg_dir, "exporter_main_v2.py"))
except ImportError:
    pass

exporter_script = None
for path in exporter_candidates:
    if os.path.exists(path):
        exporter_script = path
        break

if not exporter_script:
    print("ERROR: Could not find exporter_main_v2.py")
    sys.exit(1)

# Find latest checkpoint
import re
ckpt_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("ckpt-")]
if not ckpt_files:
    print(f"ERROR: No checkpoints found in {CHECKPOINT_DIR}")
    sys.exit(1)

ckpt_nums = set()
for f in ckpt_files:
    match = re.search(r'ckpt-(\d+)', f)
    if match:
        ckpt_nums.add(int(match.group(1)))

latest_ckpt = max(ckpt_nums)
print(f"Using checkpoint: ckpt-{latest_ckpt}")

cmd = [
    sys.executable, exporter_script,
    "--input_type", "image_tensor",
    "--pipeline_config_path", PIPELINE_CONFIG,
    "--trained_checkpoint_dir", CHECKPOINT_DIR,
    "--output_directory", EXPORT_DIR,
]

subprocess.run(cmd, check=True)
print(f"SavedModel exported to: {EXPORT_DIR}/saved_model/")

# ─── STEP 2: CONVERT TO TFLITE WITH INT8 QUANTIZATION ───────────────────────

print("\nStep 2: Converting to TFLite with int8 quantization...")


def representative_dataset_gen():
    """Generate representative dataset for quantization calibration."""
    dataset = tf.data.TFRecordDataset(TRAIN_TFRECORD)

    count = 0
    for raw_record in dataset:
        if count >= NUM_CALIBRATION_IMAGES:
            break

        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        # Decode image
        img_bytes = example.features.feature["image/encoded"].bytes_list.value[0]
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, [INPUT_SIZE, INPUT_SIZE])
        img = tf.cast(img, tf.uint8)
        img = tf.expand_dims(img, 0)

        yield [img]
        count += 1

    print(f"  Used {count} images for calibration")


# Load SavedModel and convert
saved_model_dir = os.path.join(EXPORT_DIR, "saved_model")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Full integer quantization for Coral TPU
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

with open(TFLITE_OUTPUT, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved to: {TFLITE_OUTPUT}")
print(f"Model size: {os.path.getsize(TFLITE_OUTPUT) / 1024 / 1024:.1f} MB")

# ─── STEP 3: EDGE TPU COMPILATION INSTRUCTIONS ──────────────────────────────

print(f"""
{'='*60}
NEXT: Compile for Edge TPU
{'='*60}

The Edge TPU compiler only runs on Linux (or Colab).
Transfer {TFLITE_OUTPUT} to your Raspberry Pi or a Linux machine, then:

  # Install compiler (Debian/Ubuntu):
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
  sudo apt update
  sudo apt install edgetpu-compiler

  # Compile:
  edgetpu_compiler {TFLITE_OUTPUT}

This produces model_edgetpu.tflite — that's what you load on the Pi with pycoral.

Alternatively, use Google Colab:
  !apt install edgetpu-compiler
  !edgetpu_compiler model.tflite
""")
