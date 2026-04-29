"""
post_processing/edge_mode.py — Adaptive Edge Deployment Mode

Implements Section IV-K of the paper:

  "For deployment on resource-constrained edge hardware — embedded security
   processors, NVIDIA Jetson Nano, or Raspberry Pi platforms — the Adaptive
   Edge Deployment module monitors GPU memory utilization and inference latency
   in real time. When available GPU memory falls below 2 GB or frame inference
   latency exceeds 40 ms, the module automatically switches the active model
   from YOLOv8s to the lighter YOLOv8n variant and reduces input resolution
   from 640×640 to 512×512 pixels. When resource availability recovers, the
   system seamlessly switches back to YOLOv8s."

Two switching triggers (OR logic):
  1. Inference latency > 40 ms for N consecutive frames  (latency_trigger)
  2. Available GPU VRAM < 2048 MB                        (memory_trigger)

Recovery requires latency < 30 ms sustained for recovery_window frames
AND GPU VRAM > 3072 MB.

Model variants:
  full : yolov8s.pt, imgsz=640   (best accuracy)
  edge : yolov8n.pt, imgsz=512   (lightweight, resource-friendly)
"""

from collections import deque
from typing import Dict, Optional

# Try to import torch for GPU memory querying
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _get_gpu_free_mb() -> Optional[float]:
    """Return free GPU VRAM in MB, or None if CUDA not available."""
    if not _TORCH_AVAILABLE:
        return None
    try:
        if not _torch.cuda.is_available():
            return None
        # reserved - allocated = currently free within PyTorch's cache
        free = _torch.cuda.mem_get_info()[0]   # bytes free on device
        return free / (1024 * 1024)
    except Exception:
        return None


class EdgeModeManager:
    """
    Adaptive Edge Deployment Mode manager.

    Monitors both inference latency and GPU memory, automatically switching
    between YOLOv8s (full-accuracy) and YOLOv8n (lightweight edge) modes.

    Parameters
    ----------
    full_model_path : str
        Path to the full-accuracy model (default: yolov8s.pt).
    edge_model_path : str
        Path to the edge-mode lightweight model (default: yolov8n.pt).
    latency_trigger_ms : float
        Inference latency threshold to enter edge mode (paper: 40ms).
    latency_trigger_frames : int
        How many consecutive high-latency frames before switching.
    memory_trigger_mb : float
        Free GPU VRAM threshold to enter edge mode (paper: 2048 MB = 2 GB).
    recovery_latency_ms : float
        Latency below which recovery is counted (paper: < 30ms sustained).
    recovery_memory_mb : float
        Free GPU VRAM above which recovery is allowed (3072 MB = 3 GB).
    recovery_window : int
        Consecutive low-latency frames required to return to full mode.
    """

    FULL_IMGSZ = 640
    EDGE_IMGSZ = 512  # Paper: "reduces input resolution from 640×640 to 512×512"

    def __init__(
        self,
        full_model_path: str = "yolov8s.pt",
        edge_model_path: str = "yolov8n.pt",
        latency_trigger_ms: float = 40.0,
        latency_trigger_frames: int = 5,
        memory_trigger_mb: float = 2048.0,
        recovery_latency_ms: float = 30.0,
        recovery_memory_mb: float = 3072.0,
        recovery_window: int = 15,
    ):
        self.full_model_path = full_model_path
        self.edge_model_path = edge_model_path
        self.latency_trigger_ms = latency_trigger_ms
        self.latency_trigger_frames = latency_trigger_frames
        self.memory_trigger_mb = memory_trigger_mb
        self.recovery_latency_ms = recovery_latency_ms
        self.recovery_memory_mb = recovery_memory_mb
        self.recovery_window = recovery_window

        self._current_mode: str = "full"          # "full" | "edge"
        self._high_latency_count: int = 0         # consecutive high-latency frames
        self._recovery_counter: int = 0           # consecutive recovery frames
        self._latency_history: deque = deque(maxlen=100)
        self._memory_history: deque = deque(maxlen=50)
        self._last_switch_reason: str = ""

    @property
    def current_mode(self) -> str:
        return self._current_mode

    def _should_enter_edge(self, latency_ms: float, free_mb: Optional[float]) -> bool:
        """Return True if edge mode should be triggered."""
        # Trigger 1: sustained high latency
        if latency_ms > self.latency_trigger_ms:
            self._high_latency_count += 1
        else:
            self._high_latency_count = 0

        if self._high_latency_count >= self.latency_trigger_frames:
            self._last_switch_reason = (
                f"latency={latency_ms:.0f}ms > {self.latency_trigger_ms:.0f}ms "
                f"for {self._high_latency_count} frames"
            )
            return True

        # Trigger 2: GPU memory pressure
        if free_mb is not None and free_mb < self.memory_trigger_mb:
            self._last_switch_reason = (
                f"GPU free={free_mb:.0f}MB < {self.memory_trigger_mb:.0f}MB"
            )
            return True

        return False

    def _should_recover(self, latency_ms: float, free_mb: Optional[float]) -> bool:
        """Return True if recovery to full mode conditions are met."""
        memory_ok = (free_mb is None) or (free_mb >= self.recovery_memory_mb)
        latency_ok = latency_ms < self.recovery_latency_ms

        if latency_ok and memory_ok:
            self._recovery_counter += 1
        else:
            self._recovery_counter = 0

        return self._recovery_counter >= self.recovery_window

    def check_and_adapt(self, current_latency: float) -> Dict:
        """
        Evaluate current system resources and return adaptation decision.

        Parameters
        ----------
        current_latency : float
            Most recent frame inference latency in milliseconds.

        Returns
        -------
        dict with keys:
            model_variant : str | None
                Path to model to load (None = no change needed).
            input_size    : int
                Recommended input resolution (640 or 512).
            mode_changed  : bool
                True if mode switched this call.
            current_mode  : str
                "full" or "edge".
            switch_reason : str
                Human-readable explanation of last switch event.
            gpu_free_mb   : float | None
                Current free GPU VRAM in MB (None on CPU-only systems).
        """
        self._latency_history.append(current_latency)
        free_mb = _get_gpu_free_mb()
        if free_mb is not None:
            self._memory_history.append(free_mb)

        mode_changed = False
        model_variant = None  # None = no model swap needed

        if self._current_mode == "full":
            if self._should_enter_edge(current_latency, free_mb):
                self._current_mode = "edge"
                self._recovery_counter = 0
                mode_changed = True
                model_variant = self.edge_model_path
                print(
                    f"[EdgeMode] full → edge | reason: {self._last_switch_reason} "
                    f"| imgsz: {self.FULL_IMGSZ} → {self.EDGE_IMGSZ} "
                    f"| model: {self.full_model_path} → {self.edge_model_path}"
                )

        elif self._current_mode == "edge":
            if self._should_recover(current_latency, free_mb):
                self._current_mode = "full"
                self._high_latency_count = 0
                self._recovery_counter = 0
                mode_changed = True
                model_variant = self.full_model_path
                print(
                    f"[EdgeMode] edge → full | latency={current_latency:.0f}ms recovered "
                    f"| imgsz: {self.EDGE_IMGSZ} → {self.FULL_IMGSZ} "
                    f"| model: {self.edge_model_path} → {self.full_model_path}"
                )

        input_size = self.FULL_IMGSZ if self._current_mode == "full" else self.EDGE_IMGSZ

        return {
            "model_variant": model_variant,
            "input_size":    input_size,
            "mode_changed":  mode_changed,
            "current_mode":  self._current_mode,
            "switch_reason": self._last_switch_reason,
            "gpu_free_mb":   free_mb,
        }

    def get_stats(self) -> Dict:
        """Return current edge mode statistics for the UI status panel."""
        lat_history = list(self._latency_history)
        mem_history = list(self._memory_history)
        avg_lat = round(sum(lat_history) / len(lat_history), 2) if lat_history else 0.0
        avg_mem = round(sum(mem_history) / len(mem_history), 1) if mem_history else None
        return {
            "current_mode":        self._current_mode,
            "avg_latency_ms":      avg_lat,
            "last_latency_ms":     round(lat_history[-1], 2) if lat_history else 0.0,
            "gpu_free_mb":         round(mem_history[-1], 1) if mem_history else None,
            "avg_gpu_free_mb":     avg_mem,
            "recovery_counter":    self._recovery_counter,
            "high_latency_count":  self._high_latency_count,
            "last_switch_reason":  self._last_switch_reason,
            "latency_trigger_ms":  self.latency_trigger_ms,
            "memory_trigger_mb":   self.memory_trigger_mb,
        }
