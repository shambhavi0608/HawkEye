# Custom Weapon Dataset

This repository utilizes a securely prepared custom dataset (25,000 annotated images).
Because of the sensitive nature of weapon images, security guidelines, and file-size constraints, the raw dataset is **not published in the root repository**.

## Dataset Statistics:
* **Total Images:** ~25,000
* **Classes:** 
  - `Handgun`
  - `Knife`
  - `Rifle`
  - `Shotgun`
* **Train/Val/Test Split:** 70% / 20% / 10%

## Reproducibility
To reproduce the paper's claimed **mAP@50 of 0.961**:
1. Acquire a high-quality annotated weapon dataset structured in YOLO format.
2. Place the images/labels under `data/images` and `data/labels`.
3. Update `data/custom_dataset.yaml` paths if necessary.
4. Run the official training script: `python scripts/train_yolov8s.py` 
5. Replace the generated `weights/best.pt` over the `weapon_model.pt` in the root and run `app.py`.
