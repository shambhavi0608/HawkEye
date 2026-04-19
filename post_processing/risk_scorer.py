"""
post_processing/risk_scorer.py -- Context-Aware Risk Scoring

Risk is derived from:
- effective confidence when scene filtering has adjusted it, otherwise raw confidence
- relative box area
- spatial priority (ROI / central bias)
- class severity so long guns rank above small blades at the same confidence
"""

from typing import Dict, List, Tuple


class RiskScorer:
    """
    Context-aware risk scorer combining confidence, area, and spatial priority.

    Parameters
    ----------
    w1, w2, w3 : float
        Weights for Cs, As, Ps components respectively.
    """

    def __init__(self, w1: float = 0.5, w2: float = 0.3, w3: float = 0.2):
        assert abs(w1 + w2 + w3 - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.class_severity = {
            "Shotgun": 1.0,
            "Rifle": 0.95,
            "Handgun": 0.85,
            "Knife": 0.65,
        }

    def _compute_area_score(self, bbox: List[int], frame_shape: Tuple[int, int]) -> float:
        """Normalized bounding box area relative to frame area."""
        x1, y1, x2, y2 = bbox
        bbox_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        frame_h, frame_w = frame_shape[:2]
        frame_area = frame_h * frame_w
        if frame_area == 0:
            return 0.0
        return min(1.0, bbox_area / frame_area)

    def _compute_spatial_priority(
        self,
        bbox: List[int],
        frame_shape: Tuple[int, int],
        roi_zones: List | None,
        in_roi: bool = False,
    ) -> float:
        """
        Spatial priority score:
          1.0 if centroid is in ROI or in central 50% of frame
          0.5 otherwise
        """
        if in_roi:
            return 1.0

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        frame_h, frame_w = frame_shape[:2]

        # Check if centroid is in central 50% of frame (25%–75% in each dimension)
        in_center_x = (frame_w * 0.25) <= cx <= (frame_w * 0.75)
        in_center_y = (frame_h * 0.25) <= cy <= (frame_h * 0.75)
        return 1.0 if (in_center_x and in_center_y) else 0.5

    @staticmethod
    def get_risk_level(risk_score: float) -> str:
        """Convert numeric risk score to categorical risk level."""
        if risk_score >= 0.60:
            return "High"
        elif risk_score >= 0.40:
            return "Medium"
        return "Low"

    def score(
        self,
        detection: Dict,
        frame_shape: Tuple[int, int],
        roi_zones: List | None = None,
        in_roi: bool = False,
    ) -> Dict:
        """
        Compute risk score and level for a detection.

        Parameters
        ----------
        detection : dict
            Must have keys: confidence, bbox.
        frame_shape : tuple
            (height, width[, channels]) of the frame.
        roi_zones : list, optional
            List of ROI polygons (not used directly here; ROIMonitor handles this).
        in_roi : bool
            Whether detection centroid is inside a defined ROI zone.

        Returns
        -------
        dict
            {risk_score: float, risk_level: str}
        """
        cs = float(detection.get("effective_confidence", detection.get("confidence", 0.0)))
        bbox = detection.get("bbox", [0, 0, 10, 10])
        class_name = detection.get("class_name", "")

        as_score = self._compute_area_score(bbox, frame_shape)
        ps = self._compute_spatial_priority(bbox, frame_shape, roi_zones, in_roi)
        severity = self.class_severity.get(class_name, 0.75)

        r = self.w1 * cs + self.w2 * as_score + self.w3 * ps
        r *= 0.65 + (0.35 * severity)
        r = round(min(1.0, r), 4)
        level = self.get_risk_level(r)

        return {
            "risk_score": r,
            "risk_level": level,
            "risk_components": {
                "confidence": round(cs, 4),
                "area": round(as_score, 4),
                "spatial_priority": round(ps, 4),
                "class_severity": round(severity, 4),
            },
        }
