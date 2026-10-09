from __future__ import annotations

import unittest

import cv2
import numpy as np

from object_recognition.image_ops import (
    analysis_gray,
    decode_image,
    encode_jpeg,
    merge_side_by_side,
    normalized_difference,
    sharpness_score,
)


def checkerboard(width: int, height: int, offset: int = 0) -> np.ndarray:
    yy, xx = np.indices((height, width))
    board = (((xx // 12 + yy // 12 + offset) % 2) * 180 + 35).astype(np.uint8)
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)


class ImageOpsTests(unittest.TestCase):
    def test_jpeg_round_trip_and_sharpness(self) -> None:
        image = checkerboard(320, 240)
        decoded = decode_image(encode_jpeg(image))
        self.assertEqual(decoded.shape, image.shape)
        self.assertGreater(sharpness_score(decoded), 100)

    def test_difference_uses_decoded_pixels(self) -> None:
        first = analysis_gray(checkerboard(320, 240, 0), 0.6, 160)
        same = analysis_gray(checkerboard(320, 240, 0), 0.6, 160)
        changed = analysis_gray(checkerboard(320, 240, 1), 0.6, 160)
        self.assertEqual(normalized_difference(first, same), 0)
        self.assertGreater(normalized_difference(first, changed), 0.5)

    def test_central_roi_ignores_border_only_background_motion(self) -> None:
        first = np.full((200, 300, 3), 100, dtype=np.uint8)
        changed_border = first.copy()
        changed_border[:30, :] = 255
        changed_border[-30:, :] = 0
        before = analysis_gray(first, 0.6, 160)
        after = analysis_gray(changed_border, 0.6, 160)
        self.assertEqual(normalized_difference(before, after), 0)

    def test_merge_preserves_order_and_aspect_ratio(self) -> None:
        front = np.full((200, 100, 3), (0, 0, 255), dtype=np.uint8)
        back = np.full((100, 200, 3), (255, 0, 0), dtype=np.uint8)
        merged = merge_side_by_side(front, back, max_height=200, max_width=1000)
        self.assertEqual(merged.shape[:2], (200, 500))
        self.assertGreater(float(merged[:, :100, 2].mean()), 240)
        self.assertGreater(float(merged[:, 100:, 0].mean()), 240)


if __name__ == "__main__":
    unittest.main()
