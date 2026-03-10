"""
STEP 9: Run inference on Coral Edge TPU (on Raspberry Pi).

Usage: python inference.py

This runs on the Raspberry Pi with the Coral USB Accelerator connected.
"""

import time
import cv2
import numpy as np
from pycoral.adapters import common, detect
from pycoral.utils.edgetpu import make_interpreter

# ─── CONFIG ──────────────────────────────────────────────────────────────────

MODEL_PATH = "model_edgetpu.tflite"
LABEL_MAP = {1: "stop", 2: "yield", 3: "no_entry", 4: "one_way"}
CONFIDENCE_THRESHOLD = 0.5
INPUT_SIZE = 320

# ─── SETUP ───────────────────────────────────────────────────────────────────

# Initialize Edge TPU interpreter
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()

# Initialize camera (PiCar-X camera)
from picamera2 import Picamera2

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size": (640, 480)}))
cam.start()
time.sleep(2)

print("Running inference... Press Ctrl+C to stop.")

# ─── INFERENCE LOOP ──────────────────────────────────────────────────────────

try:
    while True:
        frame = cam.capture_array()

        # Preprocess
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))

        # Run inference
        common.set_input(interpreter, img_resized)

        start = time.perf_counter()
        interpreter.invoke()
        inference_ms = (time.perf_counter() - start) * 1000

        # Get detections
        objs = detect.get_objects(interpreter, CONFIDENCE_THRESHOLD)

        # Scale boxes back to original frame size
        scale_x = frame.shape[1] / INPUT_SIZE
        scale_y = frame.shape[0] / INPUT_SIZE

        for obj in objs:
            bbox = obj.bbox
            x1 = int(bbox.xmin * scale_x)
            y1 = int(bbox.ymin * scale_y)
            x2 = int(bbox.xmax * scale_x)
            y2 = int(bbox.ymax * scale_y)

            label = LABEL_MAP.get(obj.id + 1, f"class_{obj.id}")
            score = obj.score

            print(f"  {label}: {score:.2f} at [{x1},{y1},{x2},{y2}]")

            # Draw on frame (optional, for debugging)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {score:.0%}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"Inference: {inference_ms:.1f}ms | Detections: {len(objs)}")

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    cam.stop()
