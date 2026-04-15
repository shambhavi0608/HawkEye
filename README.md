# Sentinel Alpha

Flask-based weapon detection demo with live webcam, image upload, video processing,
ROI polygons, evidence logging, low-light CLAHE preprocessing, and a multi-stage
post-processing pipeline.

## Current repo status

This repository currently ships:

- `yolov8s.pt` as the primary general YOLO model
- `weapon_model.pt` as an auxiliary Hugging Face checkpoint when present
- `yolov8n.pt` for scene-aware person detection

This repository does not currently ship:

- a bundled custom training dataset
- a validated custom-trained YOLOv8s checkpoint for 4 weapon classes
- reproducible benchmark reports for `96.1%` accuracy or `mAP@50`
- Jetson Nano plus TensorRT deployment artifacts

## What is working well

- Flask app structure and multi-page UI
- image, video, and webcam inference flows
- CLAHE preprocessing for dark frames
- ROI polygon drawing and ROI-based gating
- evidence logging and cooldown flow
- dual-engine inference path with primary plus auxiliary model loading

## Important limitations

- Runtime detections are generated from the models bundled in the repo, not from a
  documented custom training run.
- Confidence values are raw model confidences and should not be presented as paper
  accuracy metrics.
- Post-processing modules are useful application heuristics, but the repo does not
  include ablation studies or statistical proof for false-positive reduction claims.
- Edge mode currently adjusts inference resolution only; TensorRT export/deployment
  is not included.

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Flask app:

```bash
python app.py
```

4. Open [http://localhost:5000](http://localhost:5000)

## Training utilities

The `scripts/` folder contains helper utilities for:

- merging user-supplied YOLO datasets into a unified 4-class dataset
- running fine-tuning with Ultralytics YOLO on a user-supplied `data.yaml`

These scripts are scaffolding only. They require your own dataset and do not prove
any benchmark claim by themselves.
