"""OpenCV/NumPy image decoding, scoring, comparison, and merging."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def decode_image(data: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    """Return a BGR image from JPEG bytes or an existing OpenCV array."""

    if isinstance(data, np.ndarray):
        image = data
    else:
        encoded = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("유효한 JPEG 이미지를 디코딩할 수 없습니다.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("지원하지 않는 이미지 배열 형식입니다.")
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(image)


def read_jpeg(path: str | Path) -> bytes:
    path = Path(path)
    data = path.read_bytes()
    decode_image(data)
    return data


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg", decode_image(image), [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )
    if not ok:
        raise ValueError("JPEG 인코딩에 실패했습니다.")
    return encoded.tobytes()


def central_roi(image: np.ndarray, fraction: float) -> np.ndarray:
    image = decode_image(image)
    height, width = image.shape[:2]
    roi_height = max(1, round(height * fraction))
    roi_width = max(1, round(width * fraction))
    top = (height - roi_height) // 2
    left = (width - roi_width) // 2
    return image[top : top + roi_height, left : left + roi_width]


def analysis_gray(image: np.ndarray, roi_fraction: float, target_width: int) -> np.ndarray:
    roi = central_roi(image, roi_fraction)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, target_width / gray.shape[1])
    new_width = max(1, round(gray.shape[1] * scale))
    new_height = max(1, round(gray.shape[0] * scale))
    if (new_width, new_height) != (gray.shape[1], gray.shape[0]):
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return gray


def normalized_difference(first: np.ndarray, second: np.ndarray) -> float:
    """Mean absolute pixel difference in the range 0..1."""

    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("normalized_difference expects grayscale images")
    if first.shape != second.shape:
        second = cv2.resize(second, (first.shape[1], first.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.mean(cv2.absdiff(first, second)) / 255.0)


def sharpness_score(image: np.ndarray, roi_fraction: float = 1.0) -> float:
    gray = cv2.cvtColor(central_roi(image, roi_fraction), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(image: np.ndarray, roi_fraction: float = 1.0) -> float:
    gray = cv2.cvtColor(central_roi(image, roi_fraction), cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    width = max(1, round(image.shape[1] * height / image.shape[0]))
    interpolation = cv2.INTER_AREA if height < image.shape[0] else cv2.INTER_CUBIC
    return cv2.resize(image, (width, height), interpolation=interpolation)


def merge_side_by_side(
    front: np.ndarray,
    back: np.ndarray,
    max_height: int = 1200,
    max_width: int = 1800,
) -> np.ndarray:
    """Put front on the left and back on the right without aspect distortion."""

    front = decode_image(front)
    back = decode_image(back)
    target_height = min(max(front.shape[0], back.shape[0]), max_height)
    left = _resize_to_height(front, target_height)
    right = _resize_to_height(back, target_height)
    merged = np.hstack((left, right))
    if merged.shape[1] > max_width:
        scale = max_width / merged.shape[1]
        merged = cv2.resize(
            merged,
            (max_width, max(1, round(merged.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return merged


def save_jpeg(path: str | Path, image: np.ndarray, quality: int = 90) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_jpeg(image, quality))
    return path
