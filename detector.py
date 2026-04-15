# -*- coding: utf-8 -*-
"""
detector.py -- weapon detection runtime

The current runtime uses a dual-engine inference path:
- `yolov8s.pt` as the primary general detector.
- `weapon_model.pt` as an optional auxiliary Hugging Face weapon detector.

This repository does not bundle a custom-trained checkpoint or validation report.
Confidence values returned here are raw model confidences and must not be treated
as paper-level accuracy metrics.
"""

import cv2
import os
import re
import shutil
import time
import numpy as np
from ultralytics import YOLO

HF_WEAPON_REPO = "Subh775/Threat-Detection-YOLOv8n"
HF_WEAPON_FILE = "weights/best.pt"


def _normalize_weapon_label(label: str) -> str:
    clean = label.replace("_", " ").replace("-", " ")
    clean = re.sub(r"\s+", " ", clean.strip())
    return clean.title()


def _name_to_weapon(name: str):
    """Map raw YOLO class name to a weapon label, or None if not a weapon class."""
    n = name.lower().strip()
    if n in {"hand", "human", "person", "people", "bottle", "keyboard", "mouse"}:
        return None

    if any(k in n for k in ("handgun", "gun", "pistol", "revolver", "firearm")):
        return "Handgun"

    if any(k in n for k in ("rifle", "sniper", "assault", "carbine", "longgun", "ak47", "ar15", "m4", "m16")):
        return "Rifle"

    if "shotgun" in n:
        return "Shotgun"

    if any(k in n for k in ("knife", "blade", "dagger", "machete", "scissors", "sword", "cutlass", "bayonet")):
        return "Knife"

    return None

def _resolve_gun_class_id(names: dict) -> int:
    for k, v in names.items():
        if str(v).lower().strip() == "gun":
            return int(k)
    return 0

def _iou_xyxy(b1, b2) -> float:
    """Intersection over Union (IoU) of two bounding boxes."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


RISK_COLORS = {
    "High": (0, 0, 255),
    "Medium": (0, 165, 255),
    "Low": (0, 220, 90),
}


class WeaponDetector:
    """
    Runtime detector for the shipped Flask demo.

    It combines a general YOLOv8 model with an optional auxiliary weapon model
    when `weapon_model.pt` is present. The output should be treated as an
    application inference signal, not as a validated benchmark result.
    """

    def __init__(
        self,
        model_path: str | None = "yolov8s.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.40,
        input_size: int = 640,  # Restoring high-fidelity resolution for YOLOv8s
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        # Determine best device and precision
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half = self.device == "cuda"

        # Load Core Backbone (YOLOv8s)
        engine_path = os.path.abspath("weapon_model.engine")
        if os.path.exists(engine_path):
            self.model_path = engine_path
            print("[WeaponDetector] EDGE ACCELERATOR: TensorRT Engine Selected.")
        else:
            self.model_path = os.path.abspath(model_path) if model_path and os.path.isfile(model_path) else "yolov8s.pt"
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        try:
            self.model.fuse()  # Fuse conv+bn for faster inference
        except:
            pass
        
        # Load Specialized Weapon weights (fallback/auxiliary)
        self.aux_path = os.path.abspath("weapon_model.pt")
        if os.path.exists(self.aux_path):
            self.aux_model = YOLO(self.aux_path)
            self.aux_model.to(self.device)
            try:
                self.aux_model.fuse()
            except:
                pass
        else:
            self.aux_model = None
        
        self.class_names = self.model.names
        self.is_custom = True
        
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._frame_count = 0

        print(f"[WeaponDetector] Neural Core [{self.device}|imgsz:{self.input_size}]")
        if self.aux_model:
            print(f"[WeaponDetector] Dual-Engine Fusion: Active")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Improved low-light enhancement using adaptive CLAHE and Gamma correction.
        """
        # Convert to LAB to isolate luminance
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        avg_brightness = np.mean(l) / 255.0
        
        # If the frame is extremely dark (less than 15% luminance)
        if avg_brightness < 0.15:
            # Apply CLAHE to luminance channel
            l_enhanced = self._clahe.apply(l)
            
            # Apply Gamma correction for better detail retrieval (gamma < 1 to brighten)
            # Dynamic gamma based on darkness
            gamma = 0.7 if avg_brightness < 0.2 else 0.85
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            l_enhanced = cv2.LUT(l_enhanced, table)
            
            # Remerge and convert back
            lab = cv2.merge((l_enhanced, a, b))
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Optional: Subtle bilateral filter to reduce noise in dark regions
            if avg_brightness < 0.2:
                frame = cv2.bilateralFilter(frame, 5, 30, 30)
                
        return frame

    @staticmethod
    def _valid_weapon_box(bbox, frame_shape, cls_name: str) -> bool:
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return False
        h, w = frame_shape[:2]
        area_pct = (bw * bh) / (w * h) * 100.0

        # Global sanity check: exclude small noise or near-full-screen detections
        if area_pct < 0.05 or area_pct > 85.0:
            return False

        aspect = bw / (bh + 1e-6)

        if cls_name == "Knife":
            # Knives are typically elongated.
            if aspect > 10.0 or aspect < 0.1:
                return True 
            if 0.6 < aspect < 1.6:
                if area_pct > 15.0: return False
                
        if cls_name in {"Handgun", "Gun", "Rifle", "Shotgun"}:
            if aspect > 12.0 or aspect < 0.05:
                return False

        return True

    def _run_model(self, model, frame, conf: float, imgsz: int, class_ids: list[int] | None = None):
        kw = dict(
            imgsz=imgsz,
            conf=conf,
            iou=self.iou_threshold,
            half=self.half, 
            device=self.device,
            verbose=False,
        )
        if class_ids is not None:
            kw["classes"] = class_ids
        return model(frame, **kw)[0]

    def _boxes_to_detections(self, r, frame_shape: tuple, model_names: dict, threshold_map: dict | None = None) -> list:
        out = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
            
        h, w = frame_shape[:2]
        
        for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            cls_id = int(cls_id)
            raw = model_names.get(cls_id, "unknown")
            raw = raw if isinstance(raw, str) else str(raw)
            weapon_cls = _name_to_weapon(raw)
            if weapon_cls is None:
                continue
            
            current_min_conf = threshold_map.get(raw, self.conf_threshold) if threshold_map else self.conf_threshold
            if weapon_cls == "Knife":
                current_min_conf = max(current_min_conf, 0.35) 
            
            if float(conf) < current_min_conf:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())
            bbox = [x1, y1, x2, y2]
            
            bw, bh = (x2 - x1), (y2 - y1)
            aspect = bw / (bh + 1e-6)

            if not self._valid_weapon_box(bbox, frame_shape, weapon_cls):
                continue

            if weapon_cls == "Handgun":
                if aspect > 6.0 or aspect < 0.15:
                    weapon_cls = "Knife"
                elif (bw > w * 0.20 and aspect > 2.2):
                    weapon_cls = "Rifle"
                elif (bh > h * 0.20 and aspect < 0.45):
                    weapon_cls = "Shotgun"
            
            c = float(conf)
            
            out.append(
                {
                    "class_name": weapon_cls,
                    "confidence": round(c, 4),
                    "raw_confidence": round(c, 4),
                    "bbox": bbox,
                    "coco_name": raw,
                    "source": "neural_synthesis",
                }
            )
        return out

    # Persistent storage for Strided Inference results
    _last_heavy_dets = []

    def detect(self, frame: np.ndarray, imgsz: int | None = None) -> tuple[list, float]:
        """
        Run Optimized Neural Inference.
        Strided Inference Protocol:
        - Auxiliary (Nano) engine runs EVERY frame for low-latency firearm tracking.
        - Primary (Large) engine runs every 5th frame to maintain SOTA feature extraction.
        """
        self._frame_count += 1
        t0 = time.perf_counter()
        proc = self._preprocess(frame)
        sz = int(imgsz) if imgsz is not None else int(self.input_size)

        all_detections = []
        
        # Engine 1: Primary Large Engine (Strided)
        # Always run for high-res analysis (imgsz >= 640) which indicates still images
        run_heavy = (self._frame_count % 5 == 0) or (sz >= 600)
        
        if run_heavy:
            r_l = self._run_model(self.model, proc, conf=self.conf_threshold, imgsz=sz)
            self._last_heavy_dets = self._boxes_to_detections(r_l, proc.shape, self.model.names)
        
        all_detections.extend(self._last_heavy_dets)

        # Engine 2: Auxiliary specialized engine (Every frame Firearms recovery)
        if self.aux_model:
            r_aux = self._run_model(self.aux_model, proc, conf=0.12, imgsz=sz)
            aux_dets = self._boxes_to_detections(r_aux, proc.shape, self.aux_model.names, threshold_map={"Gun": 0.12})
            
            # Fuse with primary results
            for ad in aux_dets:
                is_dup = False
                for ld in all_detections:
                    if _iou_xyxy(ad["bbox"], ld["bbox"]) > 0.45:
                        is_dup = True
                        break
                if not is_dup:
                    all_detections.append(ad)

        for det in all_detections:
            det.setdefault("risk_score", round(float(det.get("confidence", 0.0)), 4))
            det.setdefault(
                "risk_level",
                "High" if det["risk_score"] >= 0.75 else "Medium" if det["risk_score"] >= 0.45 else "Low",
            )

        latency = (time.perf_counter() - t0) * 1000.0
        return all_detections, latency



    def draw_detections(self, frame: np.ndarray, dets: list) -> np.ndarray:
        out = frame.copy()
        for det in dets:
            cls = det.get("class_name", "Unknown")
            conf = det.get("confidence", 0.0)
            bbox = det.get("bbox", [0, 0, 100, 100])
            risk_level = det.get("risk_level", "Low")
            risk_score = det.get("risk_score", 0.0)

            x1, y1, x2, y2 = map(int, bbox)
            color = RISK_COLORS.get(risk_level, (0, 220, 90))

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"{cls} {conf:.2f} | {risk_level} R:{risk_score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.52
            thick = 1
            (tw, th), base = cv2.getTextSize(label, font, scale, thick)
            label_h = th + base + 6

            if y1 - label_h >= 0:
                lt = y1 - label_h
                ty = y1 - base - 3
            else:
                lt = y1
                ty = y1 + th + 3

            cv2.rectangle(out, (x1, lt), (x1 + tw + 8, lt + label_h), color, cv2.FILLED)
            cv2.putText(
                out, label, (x1 + 4, ty), font, scale, (0, 0, 0), thick, cv2.LINE_AA
            )

        return out

    def switch_model(self, model_path: str | None, input_size: int):
        """Edge mode: only input resolution changes; optional explicit weight path."""
        engine_fallback = os.path.abspath("weapon_model.engine")
        chosen_path = engine_fallback if os.path.exists(engine_fallback) else model_path
        if chosen_path is not None and os.path.isfile(str(chosen_path)):
            self.model = YOLO(chosen_path)
            self.model_path = os.path.abspath(chosen_path)
            if self.model_path.endswith('.engine'): print('[WeaponDetector] Edge Mode Active: TensorRT Engine.')
            self.class_names = self.model.names
            self._gun_class_id = _resolve_gun_class_id(self.class_names)
        self.input_size = input_size
        print(f"[WeaponDetector] Resolution -> {input_size}")
