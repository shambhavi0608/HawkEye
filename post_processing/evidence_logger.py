"""
post_processing/evidence_logger.py — Automated Evidence Logging

On each high-confidence confirmed detection, saves:
  - Annotated frame snapshot as PNG
  - JSON metadata: timestamp, class, confidence, risk_score, risk_level, roi_zone, session_id
  - Filename: alert_YYYY_MM_DD_HH_MM_SS_<class>_<risk>.png
Saves to evidence_logs/ directory.
"""

import os
import cv2
import json
import uuid
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Optional


EVIDENCE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evidence_logs")


class EvidenceLogger:
    """
    Saves annotated frame snapshots and JSON metadata for high-risk detections.
    """

    def __init__(self, evidence_dir: str = EVIDENCE_DIR):
        self.evidence_dir = evidence_dir
        os.makedirs(self.evidence_dir, exist_ok=True)
        print(f"[EvidenceLogger] Saving evidence to: {self.evidence_dir}")

    def log(
        self,
        frame: np.ndarray,
        detection: Dict,
        risk_result: Dict,
        session_id: str,
        roi_zone: Optional[list] = None,
        source_mode: str = "live",
    ) -> Optional[str]:
        """
        Log a high-confidence detection event.

        Parameters
        ----------
        frame : np.ndarray
            Annotated BGR frame.
        detection : dict
            Detection dict with class_name, confidence, bbox.
        risk_result : dict
            Risk scoring result with risk_score, risk_level.
        session_id : str
            Current session identifier.
        roi_zone : list, optional
            ROI zone data if applicable.

        Returns
        -------
        str or None
            Filename of saved PNG (without directory prefix), or None on failure.
        """
        try:
            now = datetime.now(timezone.utc)
            ts_str = now.strftime("%Y_%m_%d_%H_%M_%S")
            cls = detection.get("class_name", "Unknown").replace(" ", "_")
            risk_level = risk_result.get("risk_level", "Low")

            img_filename = f"alert_{ts_str}_{cls}_{risk_level}.png"
            img_path = os.path.join(self.evidence_dir, img_filename)
            cv2.imwrite(img_path, frame)

            metadata = {
                "timestamp": now.isoformat(),
                "class": detection.get("class_name"),
                "class_name": detection.get("class_name"),
                "classification": detection.get("class_name"),
                "model_class": detection.get("coco_name"),
                "confidence": detection.get("confidence"),
                "bbox": detection.get("bbox"),
                "risk_score": risk_result.get("risk_score"),
                "risk_level": risk_level,
                "risk_components": risk_result.get("risk_components"),
                "detection_id": detection.get("detection_id"),
                "source_mode": source_mode,
                "in_roi": bool(detection.get("in_roi", False)),
                "roi_zone": roi_zone,
                "session_id": session_id,
                "frame_file": img_filename,
            }
            json_filename = img_filename.replace(".png", ".json")
            json_path = os.path.join(self.evidence_dir, json_filename)
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"[EvidenceLogger] Saved: {img_filename}")
            return img_filename

        except Exception as e:
            print(f"[EvidenceLogger] Error saving evidence: {e}")
            return None

    def list_evidence(self) -> list:
        """Return list of evidence metadata dicts sorted by timestamp (newest first)."""
        entries = []
        try:
            for fname in os.listdir(self.evidence_dir):
                if fname.endswith(".json"):
                    jpath = os.path.join(self.evidence_dir, fname)
                    with open(jpath) as f:
                        data = json.load(f)
                    entries.append(data)
        except Exception as e:
            print(f"[EvidenceLogger] Error listing evidence: {e}")

        entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return entries
