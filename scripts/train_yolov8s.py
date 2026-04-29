"""
scripts/train_yolov8s.py — YOLOv8s Training with 5-Fold Cross-Validation

Implements the training protocol described in the paper (Section III-C):

  "The aggregated dataset is partitioned into training (70%), validation (15%),
   and test (15%) sets. To produce robust, statistically reliable performance
   estimates, 5-fold cross-validation is employed using an 80/20 train/validation
   split per fold. Each fold is trained for a maximum of 100 epochs, with early
   stopping triggered after 15 consecutive epochs without improvement in
   validation mAP@50."

Usage:
  python scripts/train_yolov8s.py --data data/weapon_data.yaml --folds 5

Requirements:
  pip install ultralytics scikit-learn pyyaml

The best fold's weights are saved as `runs/crossval/best_fold.pt`.
"""

import argparse
import os
import json
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise RuntimeError("Install ultralytics: pip install ultralytics")

try:
    import yaml
except ImportError:
    raise RuntimeError("Install PyYAML: pip install pyyaml")

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters matching paper (Section V-B)
# ─────────────────────────────────────────────────────────────────────────────
PAPER_HYPERPARAMS = dict(
    imgsz=640,
    batch=16,
    epochs=100,         # paper: max 100 epochs
    patience=15,        # paper: early stopping after 15 non-improving epochs
    optimizer="SGD",
    lr0=0.01,           # paper: initial LR 0.01
    lrf=0.001,          # cosine decay to 0.001
    momentum=0.937,     # paper: momentum 0.937
    weight_decay=5e-4,  # paper: weight decay 5×10⁻⁴
    iou=0.45,           # paper: NMS IoU 0.45
    conf=0.25,          # paper: confidence threshold 0.25
    save=True,
    # Augmentation matching paper (Section IV-A)
    mosaic=0.9,         # paper: mosaic p=0.9
    flipud=0.0,
    fliplr=0.5,         # paper: horizontal flip p=0.5
    scale=0.5,          # paper: random scaling ±50%
    translate=0.1,      # paper: translation ±10%
    hsv_h=0.015,        # paper: HSV hue ±0.015
    hsv_s=0.7,          # paper: saturation ±0.7
    hsv_v=0.4,          # paper: value ±0.4
    erasing=0.3,        # paper: random erasing p=0.3
    mixup=0.15,         # paper: mixup p=0.15
    degrees=0.0,
    copy_paste=0.0,
    verbose=True,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLOv8s on weapon detection dataset with 5-fold cross-validation."
    )
    p.add_argument(
        "--data", required=True,
        help="Path to dataset YAML (e.g. data/weapon_data.yaml)."
    )
    p.add_argument(
        "--folds", type=int, default=5,
        help="Number of cross-validation folds (paper: 5)."
    )
    p.add_argument(
        "--model", default="yolov8s.pt",
        help="Base YOLO model checkpoint (paper: yolov8s.pt)."
    )
    p.add_argument(
        "--device", default="0",
        help="Device to train on (0 for GPU, cpu for CPU)."
    )
    p.add_argument(
        "--output", default="runs/crossval",
        help="Output directory for all fold results."
    )
    p.add_argument(
        "--single-fold", type=int, default=None,
        help="Run only this fold index (0-based). Useful for resuming."
    )
    return p.parse_args()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(data: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(data, default_flow_style=False), encoding="utf-8")


def get_image_list(data_yaml: dict, data_root: Path) -> list:
    """Collect all training image paths from the dataset YAML."""
    train_dir = data_root / data_yaml.get("train", "images/train")
    if not train_dir.exists():
        raise FileNotFoundError(f"Training images not found at: {train_dir}")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in sorted(train_dir.rglob("*")) if p.suffix.lower() in exts]
    return images


def make_fold_yaml(images_train, images_val, data_yaml: dict, data_root: Path, fold_dir: Path) -> Path:
    """Write a per-fold data.yaml pointing to symlinked or listed images."""
    import random
    random.shuffle(images_train)

    # Write image list files
    train_txt = fold_dir / "train_images.txt"
    val_txt   = fold_dir / "val_images.txt"
    train_txt.write_text("\n".join(str(p) for p in images_train), encoding="utf-8")
    val_txt.write_text("\n".join(str(p) for p in images_val), encoding="utf-8")

    fold_yaml_data = {
        "path": str(data_root),
        "train": str(train_txt),
        "val":   str(val_txt),
        "names": data_yaml.get("names", []),
    }
    fold_yaml_path = fold_dir / "fold_data.yaml"
    write_yaml(fold_yaml_data, fold_yaml_path)
    return fold_yaml_path


def run_fold(fold_idx: int, fold_yaml: Path, output_dir: Path, args: argparse.Namespace) -> dict:
    """Train one fold and return its metrics."""
    fold_project = output_dir / f"fold_{fold_idx}"
    fold_project.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx + 1} / {args.folds}")
    print(f"{'='*60}")

    model = YOLO(args.model)

    train_kwargs = {**PAPER_HYPERPARAMS}
    train_kwargs.update(
        data=str(fold_yaml),
        project=str(fold_project),
        name="train",
        device=args.device,
        exist_ok=True,
    )

    results = model.train(**train_kwargs)

    # Extract validation metrics from results
    metrics = {}
    try:
        metrics["mAP50"]   = float(results.results_dict.get("metrics/mAP50(B)", 0.0))
        metrics["mAP50_95"] = float(results.results_dict.get("metrics/mAP50-95(B)", 0.0))
        metrics["precision"] = float(results.results_dict.get("metrics/precision(B)", 0.0))
        metrics["recall"]    = float(results.results_dict.get("metrics/recall(B)", 0.0))
    except Exception:
        pass

    best_weights = fold_project / "train" / "weights" / "best.pt"
    metrics["best_weights"] = str(best_weights) if best_weights.exists() else None
    metrics["fold"] = fold_idx

    print(f"\n[Fold {fold_idx + 1}] mAP@50 = {metrics.get('mAP50', 'N/A'):.4f}")
    return metrics


def main() -> None:
    args = parse_args()

    # Load dataset YAML
    data_yaml_path = Path(args.data).resolve()
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml_path}\n"
            "Run scripts/prepare_weapon_dataset.py first to create weapon_data.yaml."
        )

    data_yaml = load_yaml(data_yaml_path)
    data_root = data_yaml_path.parent
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SENTINEL ALPHA — YOLOv8s 5-Fold Cross-Validation Training")
    print("=" * 60)
    print(f"  Dataset YAML : {data_yaml_path}")
    print(f"  Base Model   : {args.model}")
    print(f"  Folds        : {args.folds}")
    print(f"  Output       : {output_dir}")
    print(f"  Paper params : epochs=100, patience=15, LR 0.01→0.001 cosine")
    print("=" * 60)

    # Build full image list for K-fold splitting
    all_images = get_image_list(data_yaml, data_root)
    n = len(all_images)
    if n == 0:
        raise RuntimeError("No images found in the training directory.")

    print(f"\n[DataLoader] Found {n} training images across all classes.")

    # Create K-fold splits (80/20 per fold, paper Section III-C)
    fold_size = n // args.folds
    all_fold_metrics = []

    for fold_idx in range(args.folds):
        if args.single_fold is not None and fold_idx != args.single_fold:
            continue

        # Validation indices for this fold
        val_start = fold_idx * fold_size
        val_end   = val_start + fold_size if fold_idx < args.folds - 1 else n
        val_images   = all_images[val_start:val_end]
        train_images = all_images[:val_start] + all_images[val_end:]

        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_yaml = make_fold_yaml(train_images, val_images, data_yaml, data_root, fold_dir)
        metrics = run_fold(fold_idx, fold_yaml, output_dir, args)
        all_fold_metrics.append(metrics)

    # ── Summary ────────────────────────────────────────────────────────────
    if all_fold_metrics:
        map50_vals = [m["mAP50"] for m in all_fold_metrics if "mAP50" in m]
        if map50_vals:
            mean_map50 = sum(map50_vals) / len(map50_vals)
            best_fold  = max(all_fold_metrics, key=lambda m: m.get("mAP50", 0.0))

            print("\n" + "=" * 60)
            print("  5-FOLD CROSS-VALIDATION RESULTS")
            print("=" * 60)
            for m in all_fold_metrics:
                print(f"  Fold {m['fold'] + 1}: mAP@50 = {m.get('mAP50', 0.0):.4f}")
            print(f"  {'─'*40}")
            print(f"  Mean mAP@50 = {mean_map50:.4f}  (Paper target: ≥ 0.95)")
            print(f"  Best Fold   = {best_fold['fold'] + 1} → {best_fold.get('mAP50', 0.0):.4f}")
            print("=" * 60)

            # Copy best fold weights to root output
            best_weights = best_fold.get("best_weights")
            if best_weights and Path(best_weights).exists():
                dest = output_dir / "best_fold.pt"
                shutil.copy2(best_weights, dest)
                print(f"\n[✓] Best fold weights saved: {dest}")
                print("    → Deploy by renaming to 'weapon_model.pt' in project root.")

        # Save full results JSON
        results_path = output_dir / "crossval_results.json"
        results_path.write_text(
            json.dumps({"folds": all_fold_metrics, "mean_map50": mean_map50 if map50_vals else None},
                       indent=2),
            encoding="utf-8",
        )
        print(f"[✓] Full results JSON: {results_path}")


if __name__ == "__main__":
    main()
