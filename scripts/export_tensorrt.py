#!/usr/bin/env python3
"""Export the trained YOLOv8 weapon detection model to TensorRT.

This fulfills the paper's claim of edge deployment using TensorRT on hardware 
like the Jetson Nano.

Usage:
  python scripts/export_tensorrt.py --weights runs/train/weapon_finetune/weights/best.pt --imgsz 640
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to TensorRT.")
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to the trained PyTorch model (.pt file).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export in FP16 precision for faster inference on compatible GPUs.",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Export in INT8 precision (requires calibration data).",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=4,
        help="TensorRT workspace size in GB.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    print(f"Loading model from {weights_path}...")
    model = YOLO(str(weights_path))

    print(f"Starting TensorRT export (imgsz={args.imgsz}, half={args.half}, int8={args.int8}, workspace={args.workspace}GB)...")
    
    # Export to TensorRT format
    export_path = model.export(
        format="engine",
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        workspace=args.workspace,
        device=0, # Assuming GPU for compilation
        simplify=True
    )
    
    print(f"\nModel exported successfully. TensorRT engine saved to: {export_path}")
    print("\nTo deploy this edge model in SENTINEL ALPHA, update MODEL_PATH in app.py:")
    print(f"MODEL_PATH = '{export_path}'")

if __name__ == "__main__":
    main()
