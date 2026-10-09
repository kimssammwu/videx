"""Input adapters for files, JPEG sequences, videos, and raw JPEG bytes."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import cv2

from .image_ops import encode_jpeg, read_jpeg
from .models import FramePacket


def packet_from_file(
    path: str | Path,
    camera_id: str = "local",
    frame_id: str | int | None = None,
    timestamp: float | None = None,
) -> FramePacket:
    return FramePacket(
        data=read_jpeg(path),
        camera_id=camera_id,
        frame_id=Path(path).name if frame_id is None else frame_id,
        timestamp=time.monotonic() if timestamp is None else timestamp,
    )


def packets_from_files(
    paths: Iterable[str | Path], camera_id: str = "local", fps: float = 10.0
) -> Iterator[FramePacket]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    start = time.monotonic()
    for index, path in enumerate(paths):
        yield packet_from_file(path, camera_id, index, start + index / fps)


def packets_from_jpeg_bytes(
    frames: Iterable[bytes], camera_id: str, fps: float = 10.0
) -> Iterator[FramePacket]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    start = time.monotonic()
    for index, data in enumerate(frames):
        yield FramePacket(data, camera_id, index, start + index / fps)


def packets_from_video(
    path: str | Path, camera_id: str = "video", jpeg_quality: int = 90
) -> Iterator[FramePacket]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"동영상을 열 수 없습니다: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    try:
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = timestamp_ms / 1000.0 if timestamp_ms > 0 else index / fps
            yield FramePacket(
                encode_jpeg(image, jpeg_quality), camera_id, index, timestamp
            )
            index += 1
    finally:
        capture.release()
