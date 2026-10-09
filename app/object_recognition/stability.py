"""Time-weighted motion stability with hysteresis and large-motion reset."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class StabilitySnapshot:
    stable_ratio: float
    covered_seconds: float
    progress: float
    stable: bool
    ready: bool
    current_stable: bool
    large_motion: bool
    frame_gap: bool


@dataclass(frozen=True)
class _MotionInterval:
    started_at: float
    ended_at: float
    motion: float


class TimeWeightedStability:
    """Classify recent motion by elapsed time rather than frame count.

    A single modest spike stays in the window and lowers the stable ratio. A
    large movement or a long frame gap immediately starts a new stability run.
    """

    def __init__(
        self,
        *,
        motion_threshold: float,
        reset_threshold: float,
        window_seconds: float,
        required_seconds: float,
        enter_ratio: float,
        exit_ratio: float,
        max_frame_gap: float,
    ) -> None:
        self.motion_threshold = motion_threshold
        self.reset_threshold = reset_threshold
        self.window_seconds = max(window_seconds, required_seconds)
        self.required_seconds = required_seconds
        self.enter_ratio = enter_ratio
        self.exit_ratio = exit_ratio
        self.max_frame_gap = max_frame_gap
        self._intervals: deque[_MotionInterval] = deque()
        self._stable = False

    def reset(self) -> None:
        self._intervals.clear()
        self._stable = False

    def add(self, started_at: float, ended_at: float, motion: float) -> StabilitySnapshot:
        duration = ended_at - started_at
        frame_gap = duration <= 0 or duration > self.max_frame_gap
        large_motion = motion >= self.reset_threshold
        if frame_gap or large_motion:
            self.reset()
            return StabilitySnapshot(
                stable_ratio=0.0,
                covered_seconds=0.0,
                progress=0.0,
                stable=False,
                ready=False,
                current_stable=False,
                large_motion=large_motion,
                frame_gap=frame_gap,
            )

        self._intervals.append(_MotionInterval(started_at, ended_at, motion))
        cutoff = ended_at - self.window_seconds
        while self._intervals and self._intervals[0].ended_at <= cutoff:
            self._intervals.popleft()

        covered = 0.0
        stable_time = 0.0
        for interval in self._intervals:
            overlap = max(0.0, interval.ended_at - max(interval.started_at, cutoff))
            covered += overlap
            if interval.motion <= self.motion_threshold:
                stable_time += overlap
        ratio = stable_time / covered if covered > 0 else 0.0
        enough_time = covered + 1e-9 >= self.required_seconds
        if self._stable:
            self._stable = ratio + 1e-9 >= self.exit_ratio
        else:
            self._stable = enough_time and ratio + 1e-9 >= self.enter_ratio
        ready = enough_time and ratio + 1e-9 >= self.enter_ratio
        progress = 1.0 if self.required_seconds == 0 else min(1.0, covered / self.required_seconds)
        return StabilitySnapshot(
            stable_ratio=ratio,
            covered_seconds=covered,
            progress=progress,
            stable=self._stable,
            ready=ready,
            current_stable=motion <= self.motion_threshold,
            large_motion=False,
            frame_gap=False,
        )
