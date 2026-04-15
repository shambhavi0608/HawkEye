#!/usr/bin/env python3
"""Train a YOLO weapon detection model from a user-supplied dataset.

This script is intentionally conservative:
- it does not fabricate fold metrics
- it does not claim paper-level accuracy
- it requires the caller to provide the dataset YAML
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO weapon detection model.")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the merged YOLO data YAML file.",
    )
    parser.add_argument(
        "--weights",
        default="yolov8s.pt",
        help="Base weights to fine-tune from (default: yolov8s.pt).",
    )
    parser.add_argument(
        "--project",
        default="runs/train",
        help="Training output project folder.",
    )
    parser.add_argument(
        "--name",
        default="weapon_finetune",
        help="Training experiment name.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Overwrite existing training output if present.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_path}")

    print(f"Training from weights: {args.weights}")
    print(f"Using data file: {data_path}")
    print("Starting Ultralytics training run.")

    model = YOLO(args.weights)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        patience=args.patience,
        optimizer="AdamW",
        momentum=0.937,
        weight_decay=0.0005,
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        val=True,
        plots=True,
        save=True,
    )

    metrics = getattr(results, "results_dict", {}) or {}
    if metrics:
        print("\nValidation summary from Ultralytics:")
        for key in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
            if key in metrics:
                print(f"  {key}: {metrics[key]:.4f}")
    else:
        print("\nTraining finished. Review the generated Ultralytics run directory for metrics and plots.")



if __name__ == "__main__":
    main()
