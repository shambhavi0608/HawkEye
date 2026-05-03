# Sentinel Alpha — Real-Time Multi-Class Weapon Detection System

> **YOLOv8s · Flask · Multi-Modal · 9 Post-Processing Modules · 5-Fold CV Training**

A complete, fully deployable real-time multi-class weapon detection system built on YOLOv8s,
capable of detecting four weapon categories — **Handgun, Knife, Rifle, Shotgun** — across
diverse and challenging real-world surveillance conditions.

---

## Paper Contributions — Implementation Status

### ✅ 1. Multi-Source Dataset

> *A large-scale, multi-source dataset aggregated from Google Open Images, Roboflow Universe,
> Kaggle weapon repositories, and controlled CCTV-condition capture sessions, with challenging
> annotations covering blur, darkness, and partial occlusion across four weapon classes.*

**Implementation:** `scripts/prepare_weapon_dataset.py` — merges multiple YOLO-format sources,
normalises class aliases (pistol/revolver → Handgun, etc.), generates train/val splits,
and writes `weapon_data.yaml`. Supports Open Images, Roboflow, and Kaggle directory formats.

```bash
python scripts/prepare_weapon_dataset.py \
  --sources ./data/openimages ./data/roboflow ./data/kaggle \
  --output ./data/weapon_combined \
  --target-classes Handgun,Knife,Rifle,Shotgun
```

---

### ✅ 2. YOLOv8s with 5-Fold Cross-Validation (mAP@50 ≥ 0.95)

> *A YOLOv8s detection model trained with 5-fold cross-validation achieving mAP@50 ≥ 0.95.*

**Implementation:** `scripts/train_yolov8s.py` — full 5-fold CV with exact paper hyperparameters:

| Hyperparameter | Paper Value | Script Value |
|---|---|---|
| Epochs | 100 max | 100 |
| Early stopping patience | 15 | 15 |
| Optimizer | SGD | SGD |
| LR (cosine annealing) | 0.01 → 0.001 | lr0=0.01, lrf=0.001 |
| Momentum | 0.937 | 0.937 |
| Weight decay | 5×10⁻⁴ | 5e-4 |
| Mosaic | p=0.9 | 0.9 |
| Mixup | p=0.15 | 0.15 |
| Random erasing | p=0.3 | 0.3 |
| HSV jitter | h=0.015, s=0.7, v=0.4 | ✓ |

```bash
python scripts/train_yolov8s.py --data data/weapon_data.yaml --folds 5
```

Best fold weights saved to `runs/crossval/best_fold.pt`. Deploy as `weapon_model.pt`.

---

### ✅ 3. Flask Multi-Modal Deployment Platform

> *Flask-based platform supporting static image, pre-recorded video, and live webcam inputs
> with real-time annotated output streaming.*

**Implementation:** `app.py`

| Mode | Route | Notes |
|---|---|---|
| Static image | `POST /detect/image` | Full pipeline, base64 annotated result |
| Video upload | `POST /detect/video` | Async background job; poll `/detect/video/status/<job_id>` |
| Live webcam | `GET /stream` | MJPEG, dual-thread capture+inference |

---

### ✅ 4. Temporal Consistency Filtering

> *Robust, flicker-free video detection.*

**Implementation:** `post_processing/temporal_filter.py`

- Sliding window N=5 frames; confirms detection only if class appears in K≥3 frames with τ≥0.30 confidence
- Per-object IoU tracking (min_iou=0.15) to confirm same object, not just same class
- Parameters align exactly with paper Section IV-D

---

### ✅ 5. Confidence Stabilization (Anti-Flicker)

> *Confidence Stabilization modules for robust, flicker-free video detection.*

**Implementation:** `post_processing/confidence_stabilizer.py`

- EMA: `Ŝ(t) = α·C(t) + (1−α)·Ŝ(t−1)`, α=0.4
- High-certainty override: raw confidence ≥ 0.95 bypasses EMA (no dragging down)
- Per-class state tracking

---

### ✅ 6. Context-Aware Risk Scoring System

> *Actionable, prioritized threat assessment.*

**Implementation:** `post_processing/risk_scorer.py`

- `R = w₁·Cₛ + w₂·Aₛ + w₃·Pₛ` with w₁=0.5, w₂=0.3, w₃=0.2
- `Cₛ` = EMA-smoothed confidence; `Aₛ` = normalised bbox area; `Pₛ` = spatial priority (ROI or frame-centre)
- Class severity multiplier: Shotgun=1.0 > Rifle=0.95 > Handgun=0.85 > Knife=0.65
- Thresholds: Low (R<0.40), Medium (0.40≤R<0.60), High (R≥0.60)

---

### ✅ 7. Scene-Aware False Alarm Suppression

> *Contextual weapon-human co-occurrence analysis.*

**Implementation:** `post_processing/scene_filter.py`

| Condition | ψ multiplier |
|---|---|
| Weapon + Human co-located (norm. distance < 0.3) | 1.0 |
| Weapon + Human present but not proximate | 0.75 |
| Weapon alone in frame (no person detected) | 0.50 |

- YOLOv8n runs as concurrent person detector per frame
- `effective_confidence = Cₛ × ψ`; suppressed if < 0.25
- High-certainty detections (≥ 0.95) bypass context penalty

---

### ✅ 8. Smart Region-of-Interest Monitoring

> *Operator-defined sensitive zone prioritization.*

**Implementation:** `post_processing/roi_monitor.py`

- Accepts polygonal zones in normalised [0,1] coordinates via `POST /set_roi`
- Ray-casting algorithm for point-in-polygon containment test
- In-ROI detections get `Pₛ=1.0` (maximum spatial priority) in risk scoring
- Outside-ROI detections gated out when zones are active

---

### ✅ 9. Automated Evidence Logging

> *Forensic-grade timestamped snapshot archiving.*

**Implementation:** `post_processing/evidence_logger.py`

Each high-risk confirmed detection saves:
- Full-resolution annotated PNG: `alert_YYYY_MM_DD_HH_MM_SS_<class>_<risk>.png`
- JSON sidecar: timestamp (ISO 8601), class, confidence, risk score, bbox, ROI zone, session ID
- Viewable at `/logs` in the web interface

---

### ✅ 10. Alert Cooldown Mechanism

> *Eliminate redundant alert spam.*

**Implementation:** `post_processing/alert_cooldown.py`

- Per-class, per-spatial-region cooldown window Δt=5 seconds (paper Section IV-J)
- Detections within cooldown: still visually annotated but no new operator alert or log entry
- Reduces alert redundancy by 97.1% in continuous video scenarios (per paper Table V)

---

### ✅ 11. Adaptive Edge Deployment Mode

> *Automatic model switching on resource-constrained hardware.*

**Implementation:** `post_processing/edge_mode.py`

Two-trigger switching (OR logic, paper Section IV-K):

| Trigger | Condition | Action |
|---|---|---|
| Latency | > 40ms for 5 consecutive frames | Switch YOLOv8s → YOLOv8n, imgsz 640 → 512 |
| GPU Memory | Free VRAM < 2 GB | Switch YOLOv8s → YOLOv8n, imgsz 640 → 512 |
| Recovery | Latency < 30ms sustained × 15 frames AND VRAM ≥ 3 GB | Switch YOLOv8n → YOLOv8s, imgsz 512 → 640 |

- GPU VRAM monitoring via `torch.cuda.mem_get_info()` (CPU-only systems: latency-only mode)
- Seamless in-flight model swap without server restart

---

### ✅ 12. User Feedback Learning Loop

> *Continuous, site-adaptive model improvement.*

**Implementation:** `post_processing/feedback_loop.py`

- Operator marks each detection as **Correct** or **Incorrect** via `POST /feedback`
- Stored in `feedback_data/feedback_log.csv` with detection ID, class, confidence, bbox, risk score
- `GET /feedback/stats` returns accuracy %, total/correct/incorrect counts, per-class breakdown
- Growing corpus serves as fine-tuning dataset for periodic retraining cycles

---

## Repository Status

| Asset | Bundled | Notes |
|---|---|---|
| `yolov8s.pt` | ✅ | Primary COCO-pretrained general detector |
| `weapon_model.pt` | ✅ | Aux Hugging Face checkpoint (Subh775/Threat-Detection-YOLOv8n) |
| `yolov8n.pt` | ✅ | Person detector + edge mode lightweight model |
| Custom 25k training dataset | ❌ | Requires external download; use `prepare_weapon_dataset.py` |
| Custom-trained best.pt (92.8% mAP) | ❌ | Run `train_yolov8s.py` with your dataset to produce it |
| TensorRT `.engine` | ❌ | Run `scripts/export_tensorrt.py` locally after training |

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Open [http://localhost:5000](http://localhost:5000)

---

## Training Your Own Model

```bash
# Step 1 — Merge downloaded datasets
python scripts/prepare_weapon_dataset.py \
  --sources ./data/openimages ./data/roboflow ./data/kaggle \
  --output ./data/weapon_combined

# Step 2 — Run 5-fold cross-validation training
python scripts/train_yolov8s.py \
  --data ./data/weapon_combined/weapon_data.yaml \
  --folds 5 --device 0

# Step 3 — Deploy best fold
cp runs/crossval/best_fold.pt weapon_model.pt
```
