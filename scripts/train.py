"""
STEP 7: Train the model.

Usage: python scripts/train.py

This will take several hours on CPU. You can monitor progress via TensorBoard:
    tensorboard --logdir=training/train_output
"""

import os
import subprocess
import sys


def find_train_script():
    """Find the model_main_tf2.py script in the TF models repo."""
    # Common locations
    candidates = [
        "models/research/object_detection/model_main_tf2.py",
        os.path.join(sys.prefix, "Lib", "site-packages", "object_detection", "model_main_tf2.py"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    # Try to find it via the installed package
    try:
        import object_detection
        pkg_dir = os.path.dirname(object_detection.__file__)
        script = os.path.join(pkg_dir, "model_main_tf2.py")
        if os.path.exists(script):
            return script
    except ImportError:
        pass

    return None


train_script = find_train_script()
if not train_script:
    print("ERROR: Could not find model_main_tf2.py")
    print("Make sure the TF Object Detection API is installed.")
    sys.exit(1)

print(f"Using training script: {train_script}")

PIPELINE_CONFIG = "training/pipeline.config"
MODEL_DIR = "training/train_output"

os.makedirs(MODEL_DIR, exist_ok=True)

cmd = [
    sys.executable, train_script,
    "--pipeline_config_path", PIPELINE_CONFIG,
    "--model_dir", MODEL_DIR,
    "--alsologtostderr",
]

print(f"\nStarting training...")
print(f"Pipeline: {PIPELINE_CONFIG}")
print(f"Output:   {MODEL_DIR}")
print(f"Steps:    15000")
print(f"\nTip: Open another terminal and run:")
print(f"  tensorboard --logdir={MODEL_DIR}")
print(f"\n{'='*60}\n")

subprocess.run(cmd)
