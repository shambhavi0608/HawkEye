"""
post_processing/temporal_filter.py -- Temporal Consistency Filtering

Confirms detections only when the same class reappears with a similar location
across multiple recent frames. This is stricter than class-only voting and helps
reduce one-frame flashes that happen at unrelated positions.
"""

from collections import deque
from typing import List, Dict


def _iou_xyxy(b1: List[int], b2: List[int]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class TemporalConsistencyFilter:
    """
    Filters detections using a sliding-window temporal buffer.

    Implements Section IV-D of the paper:
    A detection event is confirmed and propagated downstream only when the
    same object class appears with confidence τ >= 0.30 in at least K = 3
    frames within the last N = 5 frames. This temporal consensus criterion
    effectively suppresses single-frame spurious detections (caused by motion
    blur, lighting transients, and partial occlusion) while introducing
    negligible latency.

    Parameters
    ----------
    window_size : int
        Number of recent frames to keep (paper: N=5).
    min_hits : int
        Minimum frames a class must appear in to be confirmed (paper: K=3).
    min_confidence : float
        Minimum confidence for a frame detection to count (paper: τ=0.30).
    min_iou : float
        Minimum IoU overlap between current and past bbox to count as the
        same object (not just same class in different location).
    """

    def __init__(
        self,
        window_size: int = 5,
        min_hits: int = 3,
        min_confidence: float = 0.30,
        min_iou: float = 0.15,
    ):
        self.window_size = window_size
        self.min_hits = min_hits
        self.min_confidence = min_confidence
        self.min_iou = min_iou

        self._buffer: deque = deque(maxlen=window_size)

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update the buffer with new frame detections and return confirmed detections.

        Parameters
        ----------
        detections : list of dict
            Each dict: {class_name, confidence, bbox, ...}

        Returns
        -------
        list of dict
            Detections confirmed by temporal consistency.
        """
        frame_detections: List[Dict] = []
        for det in detections:
            cls = det["class_name"]
            conf = det.get("confidence", 0.0)
            if conf >= self.min_confidence:
                frame_detections.append(det)

        self._buffer.append(frame_detections)
        confirmed = []
        if not self._buffer:
            return confirmed

        current_frame = self._buffer[-1]
        for det in current_frame:
            hits = 1
            for previous_frame in list(self._buffer)[:-1]:
                matched = any(
                    prev.get("class_name") == det.get("class_name")
                    and _iou_xyxy(prev.get("bbox", [0, 0, 0, 0]), det.get("bbox", [0, 0, 0, 0])) >= self.min_iou
                    for prev in previous_frame
                )
                if matched:
                    hits += 1

            if hits >= self.min_hits:
                confirmed.append(det)

        return confirmed

    def reset(self):
        """Clear the buffer."""
        self._buffer.clear()
