"""
Convert COCO annotations to YOLO format and create train/val split.

Usage: python scripts/convert_to_yolo.py

Reads:  data/annotations.json + images in data/<sign_folder>/
Writes: dataset/
          images/train/  images/val/
          labels/train/  labels/val/
          dataset.yaml
"""

import json
import os
import shutil
import random

# ─── CONFIG ──────────────────────────────────────────────────────────────────

ANNOTATIONS_PATH = "data/annotations.json"
IMAGE_BASE_DIR = "data"
OUTPUT_DIR = "dataset"
VAL_SPLIT = 0.15
SEED = 42

# ─── LOAD COCO ───────────────────────────────────────────────────────────────

with open(ANNOTATIONS_PATH, "r") as f:
    coco = json.load(f)

img_lookup = {img["id"]: img for img in coco["images"]}

ann_lookup = {}
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    if img_id not in ann_lookup:
        ann_lookup[img_id] = []
    ann_lookup[img_id].append(ann)

# YOLO uses 0-indexed classes
# COCO categories have id 1-4, YOLO needs 0-3
cat_id_to_yolo = {}
cat_names = []
for i, cat in enumerate(sorted(coco["categories"], key=lambda c: c["id"])):
    cat_id_to_yolo[cat["id"]] = i
    cat_names.append(cat["name"])

print(f"Classes: {cat_names}")
print(f"Mapping: {cat_id_to_yolo}")

# ─── TRAIN/VAL SPLIT ─────────────────────────────────────────────────────────

image_ids = list(img_lookup.keys())
random.seed(SEED)
random.shuffle(image_ids)

split_idx = int(len(image_ids) * (1 - VAL_SPLIT))
train_ids = image_ids[:split_idx]
val_ids = image_ids[split_idx:]

print(f"\nTotal: {len(image_ids)} | Train: {len(train_ids)} | Val: {len(val_ids)}")

# ─── CREATE DIRECTORIES ─────────────────────────────────────────────────────

for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

# ─── CONVERT AND COPY ───────────────────────────────────────────────────────

def convert_image(img_id, split):
    img_info = img_lookup[img_id]
    src_path = os.path.join(IMAGE_BASE_DIR, img_info["file_name"])

    if not os.path.exists(src_path):
        print(f"  WARNING: {src_path} not found")
        return False

    # Copy image
    img_filename = os.path.basename(img_info["file_name"])
    dst_img = os.path.join(OUTPUT_DIR, "images", split, img_filename)
    shutil.copy2(src_path, dst_img)

    # Convert annotations to YOLO format
    img_w = img_info["width"]
    img_h = img_info["height"]
    annotations = ann_lookup.get(img_id, [])

    label_filename = os.path.splitext(img_filename)[0] + ".txt"
    dst_label = os.path.join(OUTPUT_DIR, "labels", split, label_filename)

    with open(dst_label, "w") as f:
        for ann in annotations:
            x, y, w, h = ann["bbox"]  # COCO: top-left x, y, width, height

            # Convert to YOLO: center_x, center_y, width, height (normalized)
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            # Clamp to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))

            yolo_class = cat_id_to_yolo[ann["category_id"]]
            f.write(f"{yolo_class} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    return True


count = {"train": 0, "val": 0}

for img_id in train_ids:
    if convert_image(img_id, "train"):
        count["train"] += 1

for img_id in val_ids:
    if convert_image(img_id, "val"):
        count["val"] += 1

print(f"\nConverted — Train: {count['train']} | Val: {count['val']}")

# ─── CREATE dataset.yaml ────────────────────────────────────────────────────

yaml_content = f"""path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val: images/val

nc: {len(cat_names)}
names: {cat_names}
"""

yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"\ndataset.yaml saved to: {yaml_path}")
print("\nDone! Ready for YOLOv8 training.")
