"""
Train YOLOv8 for sign detection.

Usage: python scripts/train_yolo.py
"""

from ultralytics import YOLO


def main():
    DATASET_YAML = "dataset/dataset.yaml"
    MODEL = "yolov8n.pt"
    EPOCHS = 100
    IMG_SIZE = 320
    BATCH_SIZE = 16
    PATIENCE = 20

    model = YOLO(MODEL)

    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=0,
        workers=0,             # set to 0 on Windows to avoid multiprocessing issues
        project="runs",
        name="sign_detection",
        exist_ok=True,

        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
    )

    print(f"\nTraining complete!")
    print(f"Best model: runs/sign_detection/weights/best.pt")
    print(f"Results:    runs/sign_detection/")


if __name__ == "__main__":
    main()
