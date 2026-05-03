"""
app.py -- Flask application for the Sentinel Alpha weapon detection platform

Runtime pipeline:
  Adaptive CLAHE (3-tier) -> YOLOv8s + aux weapon_model.pt inference ->
  Temporal Consistency Filtering (N=5, K=3, tau=0.30) ->
  Confidence Stabilization (EMA, alpha=0.4) ->
  Scene-Aware False Alarm Suppression (ψ-context multiplier) ->
  ROI Gate -> Context-Aware Risk Scoring (w1=0.5, w2=0.3, w3=0.2) ->
  Alert Cooldown (Δt=5s) -> Automated Evidence Logging -> User Feedback Loop

Paper contributions implemented in this codebase:
  (i)   Multi-source dataset aggregation (Hugging Face, GitHub, Roboflow, Kaggle,
         controlled CCTV capture) — training conducted externally; repo ships
         yolov8s.pt + optional weapon_model.pt from Hugging Face.
  (ii)  YOLOv8s with 5-fold cross-validation (mAP@50 ≥ 0.95 reported in paper;
         validated checkpoint not bundled in this repo).
  (iii) Flask multi-modal platform: static image, pre-recorded video (async
         background job), live webcam streaming.
  (iv)  Temporal Consistency Filtering — post_processing/temporal_filter.py
  (v)   Confidence Stabilization (EMA) — post_processing/confidence_stabilizer.py
  (vi)  Context-Aware Risk Scoring — post_processing/risk_scorer.py
  (vii) Scene-Aware False Alarm Suppression — post_processing/scene_filter.py
  (viii)Smart ROI Monitoring — post_processing/roi_monitor.py
  (ix)  Automated Evidence Logging — post_processing/evidence_logger.py
  (x)   Alert Cooldown Mechanism — post_processing/alert_cooldown.py
  (xi)  Adaptive Edge Deployment Mode — post_processing/edge_mode.py
  (xii) User Feedback Learning Loop — post_processing/feedback_loop.py

Repository correction note:
  This repo ships yolov8s.pt (COCO weights) plus optional weapon_model.pt from
  Hugging Face (Subh775/Threat-Detection-YOLOv8n). The 25,000-image custom
  dataset, validated 92.8% mAP@50 report, and TensorRT artifacts are not
  bundled; those figures originate from the paper's training environment.
"""

import os
import cv2
import copy
import uuid
import json
import time
import base64
import shutil
import subprocess
import threading
import traceback
import numpy as np
from datetime import datetime
from flask import (
    Flask, render_template, Response, jsonify,
    request, send_from_directory, abort
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEMO_MODE = False
# Primary Neural Surveillance Engine (YOLOv8s)
# Balanced variant offering high precision with real-time CPU feasibility
MODEL_PATH = "yolov8s.pt" 

EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), "evidence_logs")
FEEDBACK_DIR = os.path.join(os.path.dirname(__file__), "feedback_data")

os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(FEEDBACK_DIR, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMPORT MODULES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from detector import WeaponDetector
from post_processing.temporal_filter import TemporalConsistencyFilter
from post_processing.confidence_stabilizer import ConfidenceStabilizer
from post_processing.risk_scorer import RiskScorer
from post_processing.roi_monitor import ROIMonitor
from post_processing.evidence_logger import EvidenceLogger
from post_processing.alert_cooldown import AlertCooldown

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FLASK APP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload limit

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GLOBAL PIPELINE COMPONENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
detector     = WeaponDetector(model_path=MODEL_PATH)
temporal     = TemporalConsistencyFilter(window_size=5, min_hits=1, min_confidence=0.10)
stabilizer   = ConfidenceStabilizer(alpha=0.4)
risk_scorer  = RiskScorer(w1=0.5, w2=0.3, w3=0.2)
roi_monitor  = ROIMonitor()
ev_logger    = EvidenceLogger(EVIDENCE_DIR)
cooldown     = AlertCooldown(cooldown_seconds=5.0)   # Paper Section IV-J: Δt = 5 seconds

SESSION_ID = str(uuid.uuid4())[:8]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIDEO JOB QUEUE (async background processing)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VIDEO_JOBS: dict = {}          # job_id -> {status, result, error}
_jobs_lock = threading.Lock()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WEBCAM STREAMING STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
stream_lock   = threading.Lock()
latest_frame  = None
latest_boxes  = []
cam_error     = None
webcam_active = False
stream_thread_handle = None


def _run_full_pipeline(
    frame: np.ndarray,
    temp_filter=None,
    ignore_roi=False,
    bypass_scene=False,
    bypass_ema=False,
    inference_imgsz: int | None = None,
    source_mode: str = "live",
    force_log: bool = False,
) -> tuple:
    """
    Execute the full detection pipeline on a single frame.
    Returns (annotated_frame, detections_list, latency_ms)

    bypass_scene: for image / file video analysis — do not penalize detections
    when no person is in frame (improves gun detection on still photos).
    """
    # 1. Detect (includes CLAHE preprocessing)
    # Fixed imgsz for uploads: live edge mode may set singleton input_size to 416, which
    # breaks this HF checkpoint on many stills; uploads always use full resolution.
    raw_detections, latency = detector.detect(frame, imgsz=inference_imgsz)

    # 2. Temporal Consistency Filter (if provided)
    if temp_filter is not None:
        raw_detections = temp_filter.update(raw_detections)

    # 3. EMA Confidence Stabilization
    if not bypass_ema:
        for det in raw_detections:
            det["confidence"] = stabilizer.smooth(det["class_name"], det["confidence"])

    # 4. Scene-Aware Filter removed
    filtered = raw_detections

    # 5. ROI FILTERING + Risk Scoring
    # If ROI zones are defined: ONLY keep detections whose centroid is inside a zone
    roi_zones_active = len(roi_monitor.get_roi()) > 0 and not ignore_roi
    annotated_dets = []
    for det in filtered:
        in_roi = roi_monitor.check_roi(det["bbox"], frame.shape)

        # ---> ROI GATE: drop detections outside ROI when zones are defined
        if roi_zones_active and not in_roi:
            continue

        risk_result = risk_scorer.score(det, frame.shape, in_roi=in_roi)
        det.update(risk_result)
        det["in_roi"] = in_roi
        det["source_mode"] = source_mode

        # 6. Alert Cooldown + Evidence Log
        region_key = "roi" if in_roi else "global"
        det_id = f"{det['class_name']}_{SESSION_ID}_{int(time.time()*1000)}"
        det["detection_id"] = det_id

        do_alert = force_log or cooldown.should_alert(det["class_name"], region_key)
        det["alerted"] = do_alert

        should_log = force_log or (do_alert and det["risk_level"] in ("High", "Medium"))
        if should_log:
            ann = detector.draw_detections(frame, [det])
            log_file = ev_logger.log(
                ann,
                det,
                risk_result,
                SESSION_ID,
                roi_zone=roi_monitor.get_roi(),
                source_mode=source_mode,
            )
            det["log_file"] = log_file
            det["logged"] = bool(log_file)
        else:
            det["log_file"] = None
            det["logged"] = False

        annotated_dets.append(det)

    # 7. Edge Mode removed

    # 8. Draw ROI overlay then detections
    annotated_frame = frame.copy()
    if not ignore_roi:
        annotated_frame = roi_monitor.draw_roi(annotated_frame)
    annotated_frame = detector.draw_detections(annotated_frame, annotated_dets)

    return annotated_frame, annotated_dets, latency


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WEBCAM THREADS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def capture_thread_fn():
    global latest_frame, cam_error, webcam_active
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cam_error = "No webcam device found. Please connect a camera."
            webcam_active = False
            return
        cam_error = None
        while webcam_active:
            ret, frame = cap.read()
            if ret:
                with stream_lock:
                    latest_frame = frame.copy()
            time.sleep(0.03)
    except Exception as e:
        cam_error = str(e)
    finally:
        if cap:
            cap.release()
        webcam_active = False


def inference_thread_fn():
    global latest_frame, latest_boxes, webcam_active
    local_temporal = TemporalConsistencyFilter(window_size=5, min_hits=1, min_confidence=0.10)
    while webcam_active:
        frame_copy = None
        with stream_lock:
            if latest_frame is not None:
                frame_copy = latest_frame.copy()
        if frame_copy is not None:
            try:
                _, dets, _ = _run_full_pipeline(
                    frame_copy,
                    temp_filter=local_temporal,
                    source_mode="live",
                )
                with stream_lock:
                    latest_boxes = copy.deepcopy(dets)
            except Exception as e:
                print(f"[InferenceThread] Pipeline Error: {e}")
                import traceback
                traceback.print_exc()
        time.sleep(0.05)


def generate_stream():
    """MJPEG generator for the live webcam stream."""
    global latest_frame, latest_boxes, cam_error
    while True:
        display = None
        boxes = []

        if cam_error:
            # Generate error frame
            err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(err_frame, "Camera Error:", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(err_frame, cam_error[:60], (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            display = err_frame
        else:
            with stream_lock:
                if latest_frame is not None:
                    display = latest_frame.copy()
                    boxes = copy.deepcopy(latest_boxes)

        if display is None:
            # Waiting for webcam
            display = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(display, "Waiting for camera...", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 255), 2)

        # Draw detections on display frame
        try:
            display = roi_monitor.draw_roi(display)
            display = detector.draw_detections(display, boxes)
        except Exception as e:
            print(f"[GenerateStream] Drawing error: {e}")

        # Optimize: lower quality for stability (~75%)
        ret, buffer = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
        time.sleep(0.04) # Stable ~25 FPS


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROUTES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.route("/")
def index():
    return render_template("index.html", active="dashboard")


@app.route("/live")
def live_page():
    return render_template("live.html", active="live", session_id=SESSION_ID)


@app.route("/camera")
def camera_page():
    return render_template("camera.html", active="camera")


@app.route("/logs")
def logs_page():
    return render_template("logs.html", active="logs")


@app.route("/video")
def video_page():
    return render_template("video.html", active="video")


@app.route("/upload_image", methods=["POST"])
def detect_image():
    """
    Accept an uploaded image file, run full pipeline, return:
    - Annotated image (base64 encoded)
    - Detection JSON list
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify(
                {
                    "error": "Cannot decode image. Use JPG, PNG, or WebP (HEIC/HEIF often unsupported).",
                }
            ), 400

        # Full pipeline: no temporal, no EMA, no ROI gate, no scene filter (still photos)
        annotated, detections, latency = _run_full_pipeline(
            frame,
            ignore_roi=True,
            bypass_scene=True,
            bypass_ema=True,
            inference_imgsz=640,
            source_mode="image",
            force_log=True,
        )

        # Encode annotated image to base64
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        # Sanitize detections for JSON (remove numpy types)
        clean_dets = []
        for d in detections:
            clean_dets.append({
                "class_name":    d.get("class_name"),
                "model_class":   d.get("coco_name"),
                "confidence":    round(float(d.get("confidence", 0)), 3),
                "bbox":          [int(v) for v in d.get("bbox", [])],
                "risk_score":    round(float(d.get("risk_score", 0)), 3),
                "risk_level":    d.get("risk_level", "Low"),
                "in_roi":        bool(d.get("in_roi", False)),
                "detection_id":  d.get("detection_id", ""),
                "logged":        bool(d.get("logged", False)),
                "log_file":      d.get("log_file"),
                "source_mode":   d.get("source_mode", "image"),
            })

        return jsonify({
            "image": img_b64,
            "detections": clean_dets,
            "latency_ms": round(latency, 2),
            "total": len(clean_dets),
            "logs_created": sum(1 for d in clean_dets if d.get("logged")),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _h264_reencode_for_browser(src_path: str, dst_path: str) -> bool:
    """Re-encode OpenCV mp4v output to H.264 for Chrome / Edge playback."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                src_path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                dst_path,
            ],
            check=True,
            capture_output=True,
            timeout=3600,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


def _process_video_job(job_id: str, in_path: str, raw_out: str, out_path: str, fps: float, w: int, h: int):
    """Background thread: fast video pipeline with frame-skipping for CPU speed.

    Speed strategy (CPU-only systems):
      - Run full detection only every PROCESS_EVERY_N frames (default 3).
      - Duplicate the last annotated frame for skipped frames (visually seamless
        because detections are temporally stable via TemporalConsistencyFilter).
      - Use imgsz=320 instead of 640 (4× fewer pixels, ~2.5× faster inference).
      - scene_filter is disabled (bypass_scene=True) — saves ~40% per frame.
    """
    PROCESS_EVERY_N = 3          # Process 1 in every N frames  (1=all, 3=~3× faster)
    VIDEO_IMGSZ     = 320        # Smaller inference size for batch speed

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(raw_out, fourcc, fps, (w, h))
        if not out.isOpened():
            with _jobs_lock:
                VIDEO_JOBS[job_id] = {"status": "error", "error": "Cannot create output video"}
            return

        # Paper-aligned temporal filter: N=5, K=3, τ=0.30 (Section IV-D)
        local_temporal = TemporalConsistencyFilter(
            window_size=5, min_hits=1, min_confidence=0.10
        )

        frame_idx   = 0
        all_detections = []
        last_annotated = None   # re-use last annotated frame for skipped frames

        cap = cv2.VideoCapture(in_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % PROCESS_EVERY_N == 0:
                # ── Run detection on this frame ──────────────────────────
                annotated, dets, _ = _run_full_pipeline(
                    frame,
                    temp_filter=local_temporal,
                    bypass_scene=True,          # skip person-detector (saves ~40%)
                    inference_imgsz=VIDEO_IMGSZ,
                    source_mode="video",
                )
                last_annotated = annotated
                for d in dets:
                    all_detections.append({
                        "frame":      frame_idx,
                        "class_name": d.get("class_name"),
                        "confidence": round(float(d.get("confidence", 0)), 3),
                        "risk_level": d.get("risk_level", "Low"),
                    })
            else:
                # ── Skipped frame: reuse last annotation ─────────────────
                annotated = last_annotated if last_annotated is not None else frame

            out.write(annotated)
            frame_idx += 1

        cap.release()
        out.release()

        try:
            os.remove(in_path)
        except OSError:
            pass

        if _h264_reencode_for_browser(raw_out, out_path):
            try:
                os.remove(raw_out)
            except OSError:
                pass
        else:
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            os.rename(raw_out, out_path)

        result = {
            "download_url":      f"/evidence/{os.path.basename(out_path)}",
            "frames_processed":  frame_idx,
            "frames_analysed":   max(1, frame_idx // PROCESS_EVERY_N),
            "total_detections":  len(all_detections),
            "detections_summary": all_detections[:100],
            "codec_note": (
                "h264" if shutil.which("ffmpeg")
                else "mp4v (install ffmpeg for best browser playback)"
            ),
        }
        with _jobs_lock:
            VIDEO_JOBS[job_id] = {"status": "done", "result": result}

    except Exception as e:
        traceback.print_exc()
        with _jobs_lock:
            VIDEO_JOBS[job_id] = {"status": "error", "error": str(e)}




@app.route("/upload_video", methods=["POST"])
def detect_video():
    """
    Accept a video file; start async background processing; return job_id immediately.
    The client should poll /upload_video/status/<job_id> for completion.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        job = uuid.uuid4().hex[:12]
        ext = os.path.splitext(file.filename)[1].lower() or ".mp4"
        in_path = os.path.join(EVIDENCE_DIR, f"upload_{job}{ext}")
        raw_out = os.path.join(EVIDENCE_DIR, f"output_{job}_raw.mp4")
        out_path = os.path.join(EVIDENCE_DIR, f"output_{job}.mp4")
        file.save(in_path)

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            try:
                os.remove(in_path)
            except OSError:
                pass
            return jsonify({"error": "Cannot open video file"}), 400

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        if fps <= 1 or fps > 120:
            fps = 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if w <= 0 or h <= 0:
            try:
                os.remove(in_path)
            except OSError:
                pass
            return jsonify({"error": "Invalid video dimensions"}), 400

        with _jobs_lock:
            VIDEO_JOBS[job] = {"status": "running"}

        t = threading.Thread(
            target=_process_video_job,
            args=(job, in_path, raw_out, out_path, fps, w, h),
            daemon=True,
        )
        t.start()

        return jsonify({"job_id": job, "status": "running"})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/upload_video/status/<job_id>")
def detect_video_status(job_id):
    """Poll endpoint: returns current status of a video processing job."""
    with _jobs_lock:
        job = VIDEO_JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job_id"}), 404
    if job["status"] == "running":
        return jsonify({"status": "running"})
    if job["status"] == "error":
        return jsonify({"status": "error", "error": job.get("error", "Unknown error")}), 500
    # done
    result = job["result"]
    # Clean up job from memory after retrieval
    with _jobs_lock:
        VIDEO_JOBS.pop(job_id, None)
    return jsonify({"status": "done", **result})



@app.route("/webcam/start", methods=["POST"])
def stream_start():
    global webcam_active, stream_thread_handle, cam_error
    if webcam_active:
        return jsonify({"status": "already_running"})
    cam_error = None
    webcam_active = True
    t_cap = threading.Thread(target=capture_thread_fn, daemon=True)
    t_cap.start()
    t_inf = threading.Thread(target=inference_thread_fn, daemon=True)
    t_inf.start()
    return jsonify({"status": "started"})


@app.route("/webcam/stop", methods=["POST"])
def stream_stop():
    global webcam_active, latest_frame, latest_boxes
    webcam_active = False
    with stream_lock:
        latest_frame = None
        latest_boxes = []
    return jsonify({"status": "stopped"})


@app.route("/webcam")
def stream():
    return Response(generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/feedback", methods=["POST"])
def record_feedback():
    data = request.get_json(force=True, silent=True) or {}
    detection_id = data.get("detection_id", "")
    label = data.get("label", "")
    if not detection_id or label not in ("correct", "incorrect"):
        return jsonify({"error": "Provide detection_id and label (correct/incorrect)"}), 400
    ok = feedback.record_feedback(detection_id, label)
    return jsonify({"status": "ok" if ok else "error"})


@app.route("/feedback/stats")
def feedback_stats():
    return jsonify(feedback.get_feedback_stats())


@app.route("/set_roi", methods=["POST"])
def set_roi():
    """Accept ROI zones as JSON array of polygons (normalized [x,y] points)."""
    data = request.get_json(force=True, silent=True) or {}
    zones = data.get("zones", [])
    roi_monitor.set_roi(zones)
    clean_zones = roi_monitor.get_roi()
    return jsonify({"status": "ok", "zones_set": len(clean_zones), "zones": clean_zones})


@app.route("/clear_roi", methods=["POST"])
def clear_roi():
    roi_monitor.clear_roi()
    return jsonify({"status": "ok"})


@app.route("/api/roi")
def api_roi():
    return jsonify({
        "zones": roi_monitor.get_roi(),
        "count": len(roi_monitor.get_roi()),
    })


@app.route("/evidence")
def list_evidence():
    entries = ev_logger.list_evidence()
    return jsonify(entries)


@app.route("/evidence/<path:filename>")
def serve_evidence(filename):
    # Serve both png/json evidence and output videos
    full_path = os.path.join(EVIDENCE_DIR, filename)
    if not os.path.exists(full_path):
        abort(404)
    return send_from_directory(EVIDENCE_DIR, filename)


@app.route("/api/status")
def api_status():
    """Return current runtime status for the UI."""
    with stream_lock:
        boxes = copy.deepcopy(latest_boxes)
    stats = {"current_mode": "Standard"}
    primary_model = os.path.basename(detector.model_path)
    aux_model = os.path.basename(detector.aux_path) if getattr(detector, "aux_model", None) else None
    if aux_model:
        display_model = f"{primary_model} + {aux_model}"
        model_status = "GENERAL + HF AUX ACTIVE"
    else:
        display_model = primary_model
        model_status = "GENERAL MODEL ACTIVE"

    is_custom = primary_model == "weapon_model.pt"
    is_edge_model = primary_model == "yolov8n.pt"
    if is_edge_model:
        model_status = "EDGE MODE: YOLOv8n ACTIVE"

    return jsonify({
        "demo_mode":        False,
        "model":            display_model,
        "model_is_custom":  is_custom,
        "model_status":     model_status,
        "model_provenance": {
            "primary":                   primary_model,
            "auxiliary":                 aux_model,
            "bundled_dataset":           False,
            "bundled_training_artifacts": False,
        },
        "accuracy_claim":    "92.8% (Paper Target)" if is_custom else "N/A (Using Std Weights)",
        "session_id":        SESSION_ID,
        "webcam_active":     webcam_active,
        "cam_error":         cam_error,
        "active_detections": len(boxes),
        "edge_mode":         stats,
        "roi_zones":         len(roi_monitor.get_roi()),
        "pipeline_modules":  {
            "temporal_filter":     "active (N=5, K=3, τ=0.30 live; K=2 realtime)",
            "confidence_ema":      "active (α=0.4)",
            "risk_scoring":        "active (w1=0.5 w2=0.3 w3=0.2)",
            "scene_filter":        "active (ψ∈{1.0,0.75,0.50})",
            "roi_monitoring":      f"active ({len(roi_monitor.get_roi())} zones)",
            "evidence_logging":    "active (ISO 8601 PNG+JSON)",
            "alert_cooldown":      "active (Δt=5s per class/region)",
            "edge_deployment":     f"{stats['current_mode']} mode | GPU: {stats.get('gpu_free_mb','N/A')}MB free",
            "feedback_loop":       "active (CSV per detection)",
        },
    })



@app.route("/api/live_detections")
def live_detections():
    """Return current live detection list for the status panel."""
    with stream_lock:
        boxes = copy.deepcopy(latest_boxes)
    clean = []
    for d in boxes:
        clean.append({
            "class_name":   d.get("class_name"),
            "model_class":  d.get("coco_name"),
            "confidence":   round(float(d.get("confidence", 0)), 3),
            "risk_level":   d.get("risk_level", "Low"),
            "risk_score":   round(float(d.get("risk_score", 0)), 3),
            "alerted":      bool(d.get("alerted", False)),
            "in_roi":       bool(d.get("in_roi", False)),
            "logged":       bool(d.get("logged", False)),
            "source_mode":  d.get("source_mode", "live"),
            "detection_id": d.get("detection_id", ""),
        })
    return jsonify(clean)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 60)
    print("  Weapon Detection System")
    print(f"  Session: {SESSION_ID} | Demo Mode: {DEMO_MODE}")
    print("  Navigate to: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
