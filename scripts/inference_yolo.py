"""
Run YOLOv8 inference on Coral Edge TPU (on Raspberry Pi).

Usage: python inference.py

Requires pycoral and the compiled _edgetpu.tflite model.
"""

import time
import cv2
import numpy as np
from pycoral.adapters import common, detect
from pycoral.utils.edgetpu import make_interpreter

# ─── CONFIG ──────────────────────────────────────────────────────────────────

MODEL_PATH = "best_full_integer_quant_edgetpu.tflite"
LABELS = {0: "stop", 1: "yield", 2: "no_entry", 3: "one_way"}
CONFIDENCE_THRESHOLD = 0.5
INPUT_SIZE = 320

# ─── SETUP ───────────────────────────────────────────────────────────────────

interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize PiCar-X camera
from picamera2 import Picamera2

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size": (640, 480)}))
cam.start()
time.sleep(2)

print("Running inference... Press Ctrl+C to stop.")
print(f"Model: {MODEL_PATH}")
print(f"Classes: {LABELS}")

# ─── INFERENCE LOOP ──────────────────────────────────────────────────────────

try:
    while True:
        frame = cam.capture_array()

        # Preprocess
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
        img_input = np.expand_dims(img_resized, axis=0).astype(np.uint8)

        # Run inference
        common.set_input(interpreter, img_resized)

        start = time.perf_counter()
        interpreter.invoke()
        inference_ms = (time.perf_counter() - start) * 1000

        # Parse YOLOv8 outputs
        # Output shape varies by export — this handles common formats
        output_data = interpreter.get_tensor(output_details[0]["index"])

        # Scale factors
        scale_x = frame.shape[1] / INPUT_SIZE
        scale_y = frame.shape[0] / INPUT_SIZE

        detections = []

        # YOLOv8 TFLite output is typically [1, num_detections, 6]
        # where each detection is [x1, y1, x2, y2, confidence, class_id]
        # OR [1, 6, num_detections] transposed
        if len(output_data.shape) == 3:
            data = output_data[0]

            # Check if transposed (6 x N instead of N x 6)
            if data.shape[0] < data.shape[1] and data.shape[0] <= 10:
                data = data.T

            for det in data:
                if len(det) >= 6:
                    x1, y1, x2, y2, conf, cls_id = det[:6]
                elif len(det) == 5 + len(LABELS):
                    # Format: [x, y, w, h, cls0_conf, cls1_conf, ...]
                    x, y, w, h = det[:4]
                    class_scores = det[4:]
                    cls_id = np.argmax(class_scores)
                    conf = class_scores[int(cls_id)]
                    x1 = x - w / 2
                    y1 = y - h / 2
                    x2 = x + w / 2
                    y2 = y + h / 2
                else:
                    continue

                if conf < CONFIDENCE_THRESHOLD:
                    continue

                # Dequantize if needed
                if output_details[0]["dtype"] == np.uint8:
                    scale, zero_point = output_details[0]["quantization"]
                    conf = (conf - zero_point) * scale

                label = LABELS.get(int(cls_id), f"class_{int(cls_id)}")
                detections.append({
                    "label": label,
                    "confidence": float(conf),
                    "bbox": (
                        int(x1 * scale_x),
                        int(y1 * scale_y),
                        int(x2 * scale_x),
                        int(y2 * scale_y),
                    ),
                })

        # Print and draw
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            print(f"  {d['label']}: {d['confidence']:.2f} at [{x1},{y1},{x2},{y2}]")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{d['label']} {d['confidence']:.0%}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"Inference: {inference_ms:.1f}ms | Detections: {len(detections)}")

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    cam.stop()
