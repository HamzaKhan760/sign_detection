"""
STEP 5: Prepare Data — Convert COCO annotations to TFRecords with train/val split.

Usage: python scripts/prepare_data.py

Reads:  data/annotations.json + images in data/<sign_folder>/
Writes: training/train/*.tfrecord, training/val/*.tfrecord
"""

import json
import os
import random
import io
import contextlib

import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util

# ─── CONFIG ──────────────────────────────────────────────────────────────────

ANNOTATIONS_PATH = "data/annotations.json"
IMAGE_BASE_DIR = "data"
TRAIN_OUTPUT = "training/train"
VAL_OUTPUT = "training/val"
VAL_SPLIT = 0.15  # 15% for validation
SEED = 42

# ─── LOAD COCO ───────────────────────────────────────────────────────────────

with open(ANNOTATIONS_PATH, "r") as f:
    coco = json.load(f)

# Build lookup: image_id -> image info
img_lookup = {img["id"]: img for img in coco["images"]}

# Build lookup: image_id -> list of annotations
ann_lookup = {}
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    if img_id not in ann_lookup:
        ann_lookup[img_id] = []
    ann_lookup[img_id].append(ann)

# Build lookup: category_id -> category name (bytes)
cat_lookup = {cat["id"]: cat["name"].encode("utf8") for cat in coco["categories"]}

# ─── TRAIN/VAL SPLIT ─────────────────────────────────────────────────────────

image_ids = list(img_lookup.keys())
random.seed(SEED)
random.shuffle(image_ids)

split_idx = int(len(image_ids) * (1 - VAL_SPLIT))
train_ids = set(image_ids[:split_idx])
val_ids = set(image_ids[split_idx:])

print(f"Total images: {len(image_ids)}")
print(f"Train: {len(train_ids)} | Val: {len(val_ids)}")

# ─── CREATE TFRECORD ─────────────────────────────────────────────────────────

def create_tf_example(img_id):
    """Create a single TFRecord example from an image and its annotations."""
    img_info = img_lookup[img_id]
    img_path = os.path.join(IMAGE_BASE_DIR, img_info["file_name"])

    if not os.path.exists(img_path):
        print(f"  WARNING: {img_path} not found, skipping")
        return None

    # Read image
    with open(img_path, "rb") as f:
        encoded_jpg = f.read()

    # Get dimensions from annotation (more reliable than re-reading)
    width = img_info["width"]
    height = img_info["height"]

    # Verify image is readable
    try:
        img = Image.open(io.BytesIO(encoded_jpg))
        width, height = img.size
    except Exception as e:
        print(f"  WARNING: can't read {img_path}: {e}")
        return None

    filename = img_info["file_name"].encode("utf8")
    image_format = b"jpg"

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text = []
    classes = []

    annotations = ann_lookup.get(img_id, [])
    for ann in annotations:
        x, y, w, h = ann["bbox"]

        # COCO bbox is [x, y, width, height] — convert to normalized coords
        xmin = x / width
        ymin = y / height
        xmax = (x + w) / width
        ymax = (y + h) / height

        # Clamp to [0, 1]
        xmin = max(0.0, min(1.0, xmin))
        ymin = max(0.0, min(1.0, ymin))
        xmax = max(0.0, min(1.0, xmax))
        ymax = max(0.0, min(1.0, ymax))

        if xmax <= xmin or ymax <= ymin:
            continue

        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        classes_text.append(cat_lookup[ann["category_id"]])
        classes.append(ann["category_id"])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        "image/height": dataset_util.int64_feature(height),
        "image/width": dataset_util.int64_feature(width),
        "image/filename": dataset_util.bytes_feature(filename),
        "image/source_id": dataset_util.bytes_feature(filename),
        "image/encoded": dataset_util.bytes_feature(encoded_jpg),
        "image/format": dataset_util.bytes_feature(image_format),
        "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
        "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
        "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
        "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
        "image/object/class/text": dataset_util.bytes_list_feature(classes_text),
        "image/object/class/label": dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def write_tfrecords(image_ids, output_dir, split_name):
    """Write a set of images to TFRecord files."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split_name}.tfrecord")
    writer = tf.io.TFRecordWriter(output_path)

    count = 0
    for img_id in image_ids:
        tf_example = create_tf_example(img_id)
        if tf_example:
            writer.write(tf_example.SerializeToString())
            count += 1

    writer.close()
    print(f"Wrote {count} examples to {output_path}")


# ─── RUN ─────────────────────────────────────────────────────────────────────

write_tfrecords(train_ids, TRAIN_OUTPUT, "train")
write_tfrecords(val_ids, VAL_OUTPUT, "val")
print("\nDone! TFRecords ready for training.")
