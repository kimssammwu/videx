"""Reusable VIDEX Flood Fill detector, independent of USB and GUI.

Frames are upright uint8 BGR arrays. Geometry and temporal defaults preserve
the original, demonstrated flood_fill_mvp implementation.
"""
from dataclasses import dataclass
import math
from numbers import Integral
import time
from typing import Literal, TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray

if __package__:
    from .outline import extract_outline
else:
    from outline import extract_outline

Image = NDArray[np.uint8]
Mask = NDArray[np.bool_]
State = Literal["WARNING", "MONITORING", "UNKNOWN"]


@dataclass(frozen=True)
class Config:
    """Pixel kernels and normalized geometry; defaults match the original MVP."""
    low: int = 50
    high: int = 150
    kernel: int = 5
    total_threshold: float = 0.18
    center_threshold: float = 0.30
    confirm: int = 3
    stale: float = 2.0
    dilation_kernel: int = 3
    roi_vertices: tuple[tuple[float, float], ...] = (
        (.36, .42), (.64, .42), (.92, .97), (.08, .97))
    center_bounds: tuple[float, float, float, float] = (.42, .45, .58, .85)
    seed_bounds: tuple[float, float, float, float] = (.46, .90, .54, .96)
    seed_target: tuple[float, float] = (.50, .94)
    seed_clearance: float = 2.0
    release_factor: float = .70
    min_image_size: int = 32
    dark_level: int = 10
    bright_level: int = 245
    extreme_fraction: float = .98
    max_barrier_density: float = .45

    def __post_init__(self) -> None:
        for name in ('low', 'high', 'kernel', 'dilation_kernel', 'confirm',
                     'min_image_size', 'dark_level', 'bright_level'):
            if not isinstance(getattr(self, name), Integral):
                raise ValueError(f'{name} must be an integer')
        if not 0 <= self.low < self.high <= 255:
            raise ValueError('Canny thresholds require 0 <= low < high <= 255')
        for size in (self.kernel, self.dilation_kernel):
            if size < 1 or size > 31 or size % 2 == 0:
                raise ValueError('kernels must be odd and between 1 and 31')
        for value in (self.total_threshold, self.center_threshold, self.release_factor,
                      self.extreme_fraction, self.max_barrier_density):
            if not 0 < value <= 1:
                raise ValueError('ratio settings must be in (0, 1]')
        if self.confirm < 1 or self.min_image_size < 1:
            raise ValueError('confirm and min_image_size must be positive')
        if not math.isfinite(self.stale) or self.stale <= 0:
            raise ValueError('stale must be finite and positive')
        if not math.isfinite(self.seed_clearance) or self.seed_clearance <= 0:
            raise ValueError('seed_clearance must be finite and positive')
        if not 0 <= self.dark_level < self.bright_level <= 255:
            raise ValueError('brightness levels require 0 <= dark < bright <= 255')
        for bounds in (self.center_bounds, self.seed_bounds):
            if len(bounds) != 4:
                raise ValueError('bounds must be (left, top, right, bottom)')
            left, top, right, bottom = bounds
            if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                raise ValueError('bounds must be ordered fractions in [0, 1]')
        vertices = np.asarray(self.roi_vertices, dtype=np.float32)
        if (vertices.shape != (4, 2) or not np.isfinite(vertices).all()
                or np.any(vertices < 0) or np.any(vertices > 1)
                or not cv2.isContourConvex(vertices)):
            raise ValueError('roi_vertices must be a convex quadrilateral in [0, 1]')
        if len(self.seed_target) != 2 or not all(0 <= p <= 1 for p in self.seed_target):
            raise ValueError('seed_target must be two fractions in [0, 1]')


class FrameAnalysis(TypedDict):
    """Arrays use the input frame's size/orientation; masks are boolean H x W."""
    image: Image
    edges: Image
    barriers: Mask
    roi: Mask
    center: Mask
    polygon: NDArray[np.int32]
    filled: Mask
    unfilled: Mask
    seed: tuple[int, int] | None
    total: float
    central: float
    edge_density: float
    invalid: str | None


@dataclass(frozen=True)
class DetectionResult:
    """Decision snapshot. Analysis arrays are borrowed; callers must not mutate.

    After a timeout, analysis contains the last frame for visualization, but
    valid is False. With no accepted frame yet, ratios and analysis are None.
    """
    state: State
    unfilled_ratio: float | None
    central_ratio: float | None
    reason: str
    frame_id: int | None
    timestamp: float | None
    updated: bool
    consecutive_hits: int
    consecutive_clears: int
    analysis: FrameAnalysis | None

    @property
    def valid(self) -> bool:
        return self.state != 'UNKNOWN'


def validate_frame(image: Image) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError('frame must be a NumPy uint8 BGR array')
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError('frame must have shape (height, width, 3) and be nonempty')


def make_regions(shape: tuple[int, int], config: Config) -> tuple[Mask, Mask, NDArray[np.int32]]:
    """Build path and center masks, using the original rounding conventions."""
    h, w = shape
    polygon = np.rint(np.array(config.roi_vertices) * [w - 1, h - 1]).astype(np.int32)
    roi_u8 = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(roi_u8, polygon, 255)
    roi = roi_u8 > 0
    center = np.zeros_like(roi)
    left, top, right, bottom = config.center_bounds
    center[round(h * top):round(h * bottom), round(w * left):round(w * right)] = True
    center &= roi
    return roi, center, polygon


def fill_regions(roi: Mask, barriers: Mask, config: Config) -> tuple[Mask, Mask, tuple[int, int] | None]:
    """Four-connected fill from a nearby bottom-center seed, bounded by ROI."""
    h, w = roi.shape
    free = roi & ~barriers
    clearance = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 3)
    seed_patch = np.zeros_like(roi)
    left, top, right, bottom = config.seed_bounds
    seed_patch[round(h * top):round(h * bottom), round(w * left):round(w * right)] = True
    ys, xs = np.where(seed_patch & (clearance >= config.seed_clearance))
    seed = None
    filled = np.zeros_like(roi)
    if len(xs):
        target_x, target_y = config.seed_target
        nearest = np.argmin((xs - w * target_x) ** 2 + (ys - h * target_y) ** 2)
        seed = (int(xs[nearest]), int(ys[nearest]))
        flood = free.astype(np.uint8)
        cv2.floodFill(flood, None, seed, 2, flags=4)
        filled = flood == 2
    return filled, roi & ~filled, seed


def analyze(image: Image, config: Config = Config()) -> FrameAnalysis:
    """Full-frame existing Canny -> closed barriers -> ROI-constrained 4-fill.

    Red includes barrier pixels as well as disconnected free pixels, and its
    denominator is the entire ROI. No contours, boxes, tracking, or depth.
    Kernel sizes are pixels at the incoming (rotation-corrected) resolution.
    """
    validate_frame(image)
    h, w = image.shape[:2]
    edges = extract_outline(image, cv2, config.low, config.high)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             np.ones((config.kernel, config.kernel), np.uint8))
    barriers = cv2.dilate(closed, np.ones((config.dilation_kernel, config.dilation_kernel), np.uint8)) > 0
    roi, center, polygon = make_regions((h, w), config)
    filled, unfilled, seed = fill_regions(roi, barriers, config)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reason = None
    if min(h, w) < config.min_image_size:
        reason = "image too small"
    elif np.mean(gray[roi] < config.dark_level) > config.extreme_fraction:
        reason = "ROI too dark / lens covered"
    elif np.mean(gray[roi] > config.bright_level) > config.extreme_fraction:
        reason = "ROI overexposed"
    elif seed is None:
        reason = "bottom-center seed blocked"
    elif np.mean(barriers[roi]) > config.max_barrier_density:
        reason = "too many edge barriers / floor texture"
    return dict(image=image, edges=edges, barriers=barriers, roi=roi,
                center=center, polygon=polygon, filled=filled, unfilled=unfilled,
                seed=seed, total=float(np.mean(unfilled[roi])),
                central=float(np.mean(unfilled[center])) if center.any() else 0.,
                edge_density=float(np.mean(barriers[roi])), invalid=reason)


class Detector:
    """One ordered stream per instance. No rotation, resizing, USB, or GUI.

    Use process(frame, timestamp, frame_id=...) for external callers. The
    existing update/tick interface remains available for the USB viewer.
    """
    def __init__(self, config: Config = Config()) -> None:
        self.config = config
        self.last_id: int | None = None
        self.last_received: float | None = None
        self.result: FrameAnalysis | None = None
        self.state: State = "UNKNOWN"
        self.reason = "waiting for LEFT frames"
        self.hits = self.clears = self.processed = 0

    def process(self, frame: Image, timestamp: float | None = None, *,
                frame_id: int | None = None, now: float | None = None) -> DetectionResult:
        """Analyze an upright BGR frame and return a decision snapshot.

        timestamp is the arrival time in seconds. Omit it to use monotonic().
        Explicit timestamps default now to that same time (supports replay).
        Pass now in the SAME clock to detect stale queued frames. An explicit
        frame_id identifies duplicates; without it, equal arrival timestamps
        identify duplicates. Pixel-identical frames with new IDs/times count.
        """
        timestamp = time.monotonic() if timestamp is None else timestamp
        if frame_id is None:
            frame_id = (self.last_id if timestamp == self.last_received
                        else (0 if self.last_id is None else self.last_id + 1))
        updated = self.update(frame, frame_id, timestamp,
                              timestamp if now is None else now)
        return self.snapshot(updated)

    def poll(self, timestamp: float | None = None) -> DetectionResult:
        """Invalidate stale results when no frames arrive; use the input clock."""
        self.tick(time.monotonic() if timestamp is None else timestamp)
        return self.snapshot(False)

    def snapshot(self, updated: bool = False) -> DetectionResult:
        r = self.result
        return DetectionResult(
            self.state, r['total'] if r else None, r['central'] if r else None,
            self.reason, self.last_id, self.last_received, updated,
            self.hits, self.clears, r)

    def unknown(self, reason: str) -> None:
        self.state, self.reason = "UNKNOWN", reason
        self.hits = self.clears = 0

    def tick(self, now: float) -> None:
        if not math.isfinite(now):
            raise ValueError('timestamp must be finite seconds')
        # Timeouts may invalidate a result; only update() evaluates images.
        if self.last_received is None or now - self.last_received > self.config.stale:
            self.unknown("no NEW LEFT frame for %.1fs" % self.config.stale)

    def update(self, image: Image, frame_id: int, received_at: float,
               now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if not isinstance(frame_id, Integral):
            raise TypeError('frame_id must be an integer')
        if not math.isfinite(received_at) or not math.isfinite(now):
            raise ValueError('timestamps must be finite seconds in the same clock')
        if frame_id != self.last_id:
            validate_frame(image)
        self.tick(now)
        if frame_id == self.last_id:
            return False
        if now - received_at > self.config.stale:
            self.unknown("received frame is already stale")
            return False
        if self.last_id is not None and frame_id < self.last_id:
            self.unknown("frame counter restarted")
        self.last_id, self.last_received = frame_id, received_at
        self.processed += 1
        self.result = r = analyze(image, self.config)
        if r["invalid"]:
            self.unknown(r["invalid"])
            return True
        candidate = (r["total"] >= self.config.total_threshold and
                     r["central"] >= self.config.center_threshold)
        clear = (r["total"] < self.config.total_threshold * self.config.release_factor or
                 r["central"] < self.config.center_threshold * self.config.release_factor)
        self.hits = self.hits + 1 if candidate else 0
        self.clears = self.clears + 1 if clear else 0
        if self.hits >= self.config.confirm:
            self.state = "WARNING"
            self.reason = "both thresholds exceeded on consecutive NEW frames"
        elif self.state == "WARNING" and self.clears < self.config.confirm:
            self.reason = "warning held; waiting for consecutive clear frames"
        else:
            self.state = "MONITORING"
            self.reason = ("candidate; waiting for consecutive NEW frames" if candidate
                           else "below warning thresholds (not proof of a clear path)")
        return True
