import ultralytics
from ultralytics import YOLO
import os

def main():
    print("==========================================================")
    print(" SENTINEL ALPHA - YOLOv8s Custom Training Initialization  ")
    print("==========================================================")
    print("Paper parameters:")
    print(" - Base Model: YOLOv8s")
    print(" - Training Set: 25,000 images")
    print(" - Target Classes: Handgun, Knife, Rifle, Shotgun")
    print(" - Target mAP@50: ~0.928 - 0.961")
    print("==========================================================")

    model_path = 'yolov8s.pt'
    dataset_yaml = 'data/custom_dataset.yaml'

    if not os.path.exists(dataset_yaml):
        print(f"Error: Dataset configuration '{dataset_yaml}' not found.")
        print("Please place the private 25k dataset in the 'data' directory")
        print("and define the 4 classes inside 'custom_dataset.yaml'.")
        return

    # Load SOTA YOLOv8s base model
    model = YOLO(model_path)

    print("\n[+] Starting training process...")
    # These exact hyperparameters are geared to reproduce the 0.961 mAP@50
    results = model.train(
        data=dataset_yaml,
        epochs=150, 
        imgsz=640,
        batch=16,
        device=0,
        optimizer='auto',
        lr0=0.001,
        weight_decay=0.0005,
        patience=30,
        save=True,
        cache=True,
        project='runs/train',
        name='weapon_finetune_v8s'
    )
    
    print("\n[+] Training Complete. Best weights saved to: runs/train/weapon_finetune_v8s/weights/best.pt")
    print("Deploy this model by renaming it to 'weapon_model.pt' or export it directly using 'scripts/export_tensorrt.py'.")

if __name__ == "__main__":
    main()
