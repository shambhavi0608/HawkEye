import os
import cv2
from ultralytics import YOLO
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEAPON_MODEL = os.path.join(BASE_DIR, "weapon_model.pt")
COCO_MODEL = os.path.join(BASE_DIR, "yolov8s.pt")
MODEL_PATH = WEAPON_MODEL if os.path.exists(WEAPON_MODEL) else COCO_MODEL
print(f"[INFO] Using model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)
class_colors = {}
img_size = 640
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=img_size, conf=0.25, iou=0.45)[0]
    names = results.names
    annotated_frame = frame.copy()
    cv2.putText(
        annotated_frame,
        f"Model: {MODEL_PATH}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        cls_id = int(cls)
        label = f"{names[cls_id]} {conf:.2f}"
        
       
        if cls_id not in class_colors:
            class_colors[cls_id] = [random.randint(0,255) for _ in range(3)]
        
        color = class_colors[cls_id]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    

    cv2.imshow("YOLOv8 Object Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
