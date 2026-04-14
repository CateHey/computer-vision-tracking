"""
Shared data classes used across the project.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Any, Dict


@dataclass
class Keypoint:
    """A single keypoint with position and confidence."""
    x: float
    y: float
    conf: float
    name: Optional[str] = None


@dataclass
class Detection:
    """A single object detection with bounding box, confidence, and optional keypoints."""
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    class_name: Optional[str] = None
    keypoints: Optional[List[Keypoint]] = None
    track_id: Optional[int] = None

    def center(self) -> Tuple[float, float]:
        """Return the (cx, cy) center of the bounding box."""
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def area(self) -> float:
        """Return the area of the bounding box."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return (x1, y1, x2, y2) tuple for IoU computation."""
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass
class FrameAnalysis:
    """Analysis result for a single frame in close-contact detection."""
    frame_idx: int
    time_sec: float
    num_detections: int
    detections: List[Detection]
    is_close: bool
    min_distance_norm: float
    max_iou: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass
class Encounter:
    """A continuous period where animals are in close contact."""
    encounter_id: int
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    num_close_frames: int

    @property
    def duration_sec(self) -> float:
        return self.end_time_sec - self.start_time_sec

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)
