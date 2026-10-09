"""Public data types for the object-recognition module."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class RecognitionState(str, Enum):
    IDLE = "IDLE"
    WAIT_FRONT = "WAIT_FRONT"
    FRONT_CAPTURED = "FRONT_CAPTURED"
    WAIT_ROTATION = "WAIT_ROTATION"
    WAIT_BACK = "WAIT_BACK"
    BACK_CAPTURED = "BACK_CAPTURED"
    MERGING = "MERGING"
    RECOGNIZING = "RECOGNIZING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class FramePacket:
    """A decoded image or JPEG payload plus transport-neutral metadata."""

    data: bytes | np.ndarray
    camera_id: str = "default"
    frame_id: str | int | None = None
    timestamp: float | None = None


@dataclass
class RecognitionConfig:
    """Thresholds are deliberately configurable for real-camera calibration."""

    roi_fraction: float = 0.60
    analysis_width: int = 160
    frame_check_interval: float = 0.10
    stable_duration: float = 1.0
    motion_threshold: float = 0.025
    rotation_threshold: float = 0.10
    duplicate_threshold: float = 0.055
    min_sharpness: float = 45.0
    min_brightness: float = 22.0
    max_brightness: float = 238.0
    phase_timeout: float = 20.0
    max_candidates: int = 30
    jpeg_quality: int = 90
    merge_max_height: int = 1200
    merge_max_width: int = 1800
    auto_acknowledge_guidance: bool = True
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        if not 0 < self.roi_fraction <= 1:
            raise ValueError("roi_fraction must be in (0, 1]")
        if self.analysis_width < 16:
            raise ValueError("analysis_width must be at least 16")
        if self.frame_check_interval < 0 or self.stable_duration < 0:
            raise ValueError("time thresholds cannot be negative")
        if not 0 <= self.motion_threshold <= 1:
            raise ValueError("motion_threshold must be in [0, 1]")
        if not 0 <= self.rotation_threshold <= 1:
            raise ValueError("rotation_threshold must be in [0, 1]")
        if not 0 <= self.duplicate_threshold <= 1:
            raise ValueError("duplicate_threshold must be in [0, 1]")
        if not 0 <= self.min_brightness < self.max_brightness <= 255:
            raise ValueError("brightness bounds must satisfy 0 <= min < max <= 255")
        if self.phase_timeout <= 0 or self.max_candidates < 1:
            raise ValueError("phase_timeout and max_candidates must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")


@dataclass(frozen=True)
class FrameMetrics:
    frame_id: str | int | None
    state: RecognitionState
    motion: float | None
    sharpness: float
    brightness: float
    side_difference: float | None
    stable: bool
    selected: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "state": self.state.value,
            "motion": self.motion,
            "sharpness": self.sharpness,
            "brightness": self.brightness,
            "side_difference": self.side_difference,
            "stable": self.stable,
            "selected": self.selected,
            "reason": self.reason,
        }
