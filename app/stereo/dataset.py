"""Raw JPEG pairs and versioned metadata shared by capture and calibration."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from .orientation import OrientationSettings, pair_orientation, prepare_pair

SCHEMA_VERSION = 2
CAMERA_MAPPING = {0: "LEFT", 1: "RIGHT"}


def decode_jpeg(jpeg, width=None, height=None):
    if not (jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")):
        raise ValueError("Invalid JPEG start/end markers")
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("JPEG could not be decoded")
    if width is not None and image.shape[:2] != (height, width):
        raise ValueError("JPEG dimensions disagree with frame metadata")
    return image


@dataclass(frozen=True)
class CapturedFrame:
    camera_id: int
    frame_id: int | None
    timestamp_us: int | None
    width: int
    height: int
    jpeg: bytes
    received_at_utc: str
    received_monotonic_ns: int
    source: str
    source_path: str | None = None

    @classmethod
    def from_wire(cls, frame, received_at=None, monotonic_ns=None):
        if frame.camera_id not in CAMERA_MAPPING:
            raise ValueError("Unknown camera_id")
        decode_jpeg(frame.jpeg, frame.width, frame.height)
        received_at = time.time() if received_at is None else received_at
        return cls(frame.camera_id, frame.frame_id, frame.timestamp_us,
                   frame.width, frame.height, frame.jpeg,
                   datetime.fromtimestamp(received_at, timezone.utc).isoformat(),
                   time.monotonic_ns() if monotonic_ns is None else monotonic_ns,
                   "usb_cdc")

    @classmethod
    def from_file(cls, path, camera_id):
        if camera_id not in CAMERA_MAPPING:
            raise ValueError("Unknown camera_id")
        path = Path(path).resolve()
        jpeg = path.read_bytes()
        image = decode_jpeg(jpeg)
        height, width = image.shape[:2]
        return cls(camera_id, None, None, width, height, jpeg,
                   datetime.now(timezone.utc).isoformat(), time.monotonic_ns(),
                   "jpeg_file_import", str(path))

    def metadata(self, filename):
        return {
            "camera_id": self.camera_id, "side": CAMERA_MAPPING[self.camera_id],
            "frame_id": self.frame_id, "timestamp_us": self.timestamp_us,
            "width": self.width, "height": self.height,
            "pc_received_at_utc": self.received_at_utc,
            "pc_received_monotonic_ns": self.received_monotonic_ns,
            "saved_path": filename, "source": self.source,
            "source_path": self.source_path,
        }


def validate_pair(left, right, orientation=None):
    if left is None or right is None:
        raise ValueError("Both LEFT and RIGHT frames are required")
    if left.camera_id != 0 or right.camera_id != 1:
        raise ValueError("Expected LEFT camera_id=0 and RIGHT camera_id=1")
    images = (decode_jpeg(left.jpeg, left.width, left.height),
              decode_jpeg(right.jpeg, right.width, right.height))
    return prepare_pair(*images, orientation)


def save_pair(output, left, right, orientation=None):
    """Publish a complete pair directory; never re-encode or overwrite a pair."""
    orientation = OrientationSettings() if orientation is None else orientation
    processed = validate_pair(left, right, orientation)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pair_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]
    destination = output / ("pair_" + pair_id)
    staging = output / (".pending_" + pair_id)
    metadata = {
        "schema_version": SCHEMA_VERSION, "pair_id": pair_id,
        "orientation": orientation.to_dict(),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "manual", "synchronized": None,
        "synchronization_verified": False, "physical_camera_mapping_verified": False,
        "timestamp_clock": "camera-local per README; firmware unverified",
        "pc_receive_delta_ms": abs(left.received_monotonic_ns - right.received_monotonic_ns) / 1e6,
        "left": left.metadata("left.jpg"), "right": right.metadata("right.jpg"),
    }
    for side, image in zip(("left", "right"), processed):
        metadata[side].update(processed_width=image.shape[1], processed_height=image.shape[0])
    staging.mkdir()
    try:
        (staging / "left.jpg").write_bytes(left.jpeg)
        (staging / "right.jpg").write_bytes(right.jpeg)
        (staging / "pair.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        staging.rename(destination)
    except Exception:
        # Only remove these three files in the directory created by this call.
        for name in ("left.jpg", "right.jpg", "pair.json"):
            (staging / name).unlink(missing_ok=True)
        staging.rmdir()
        raise
    return destination


def load_pair(manifest):
    manifest = Path(manifest).resolve()
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Pair metadata must be an object")
    if metadata.get("schema_version") not in (1, SCHEMA_VERSION):
        raise ValueError("Unsupported pair schema_version")
    if not isinstance(metadata.get("pair_id"), str) or not metadata["pair_id"]:
        raise ValueError("Missing pair_id")
    images, paths = [], []
    for side, camera_id in (("left", 0), ("right", 1)):
        item = metadata[side]
        if not isinstance(item, dict):
            raise ValueError("Frame metadata must be an object")
        if item["camera_id"] != camera_id:
            raise ValueError("Invalid camera mapping in pair metadata")
        if any(type(item.get(key)) is not int or item[key] <= 0 for key in ("width", "height")):
            raise ValueError("Invalid frame dimensions")
        if not isinstance(item.get("saved_path"), str) or not item["saved_path"]:
            raise ValueError("Missing saved image path")
        relative = Path(item["saved_path"])
        path = (manifest.parent / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(manifest.parent):
            raise ValueError("Image path must remain within its pair directory")
        image = decode_jpeg(path.read_bytes(), item["width"], item["height"])
        images.append(image)
        paths.append(path)
    processed = prepare_pair(*images, pair_orientation(metadata))
    if metadata["schema_version"] == SCHEMA_VERSION:
        for side, image in zip(("left", "right"), processed):
            item = metadata[side]
            if (item.get("processed_width"), item.get("processed_height")) != (image.shape[1], image.shape[0]):
                raise ValueError("Processed dimensions disagree with stored rotation settings")
    # Keep this API in raw-pixel space. Algorithms call prepare_pair exactly once.
    return metadata, paths, images
