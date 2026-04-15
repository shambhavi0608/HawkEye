"""
post_processing/roi_monitor.py — Smart ROI Monitoring

Accepts list of polygonal ROI zones (normalized [x,y] points).
Checks if detection bbox centroid falls inside any ROI polygon.
Elevates Ps score for in-ROI detections.
"""

import copy
import numpy as np
from typing import List, Tuple


class ROIMonitor:
    """
    Monitors whether detections fall inside user-defined polygonal ROI zones.

    Coordinates are specified in normalized [0,1] format (relative to frame size).
    """

    def __init__(self):
        self._roi_zones: List[List[List[float]]] = []  # list of polygons (list of [x,y] points)

    def set_roi(self, zones: List[List[List[float]]]):
        """
        Set the list of ROI zones.

        Parameters
        ----------
        zones : list of polygon
            Each polygon is a list of [x, y] points in normalized [0,1] coordinates.
            Example: [[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]]
        """
        normalized = []
        for zone in zones or []:
            clean_zone = []
            for point in zone:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    continue
                try:
                    x = min(1.0, max(0.0, float(point[0])))
                    y = min(1.0, max(0.0, float(point[1])))
                except (TypeError, ValueError):
                    continue
                clean_zone.append([round(x, 6), round(y, 6)])
            if len(clean_zone) >= 3:
                normalized.append(clean_zone)

        self._roi_zones = normalized

    def get_roi(self) -> List[List[List[float]]]:
        """Return current ROI zones."""
        return copy.deepcopy(self._roi_zones)

    def clear_roi(self):
        """Remove all ROI zones."""
        self._roi_zones = []

    @staticmethod
    def _point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
        """
        Ray-casting algorithm: check if a point is inside a polygon.
        All coordinates in the same space (e.g., all normalized or all pixel).
        """
        px, py = point
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi):
                inside = not inside
            j = i
        return inside

    def check_roi(self, bbox: List[int], frame_shape: Tuple[int, int]) -> bool:
        """
        Check if the centroid of a detection bbox falls inside any ROI zone.

        Parameters
        ----------
        bbox : list
            [x1, y1, x2, y2] in pixel coordinates.
        frame_shape : tuple
            (height, width[, channels]).

        Returns
        -------
        bool
            True if centroid is inside any ROI zone.
        """
        if not self._roi_zones:
            return False

        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        cx_norm = ((x1 + x2) / 2) / w
        cy_norm = ((y1 + y2) / 2) / h
        point = (cx_norm, cy_norm)

        for zone in self._roi_zones:
            if len(zone) >= 3 and self._point_in_polygon(point, zone):
                return True
        return False

    def draw_roi(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw ROI zones onto the frame for visualization.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame to draw on.

        Returns
        -------
        np.ndarray
            Frame with ROI overlays drawn.
        """
        import cv2
        frame = frame.copy()
        h, w = frame.shape[:2]

        for zone in self._roi_zones:
            pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in zone], dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        return frame
