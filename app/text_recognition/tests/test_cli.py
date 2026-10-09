from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from text_recognition.cli import main
from text_recognition.errors import InvalidImageError


class CliTests(unittest.TestCase):
    def test_success_stdout_contains_only_unicode_json(self) -> None:
        stdout = io.StringIO()
        with (
            patch("text_recognition.cli.FileImageSource") as source,
            contextlib.redirect_stdout(stdout),
        ):
            source.return_value.capture.return_value = b"jpeg"
            code = main(["--image", "unused.jpg", "--mock-text", "서울책방"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), {"text": "서울책방"})
        self.assertIn("서울책방", stdout.getvalue())

    def test_failure_is_json_with_nonzero_exit_and_not_empty_text(self) -> None:
        stdout = io.StringIO()
        with (
            patch("text_recognition.cli.FileImageSource") as source,
            contextlib.redirect_stdout(stdout),
        ):
            source.return_value.capture.side_effect = InvalidImageError("invalid")
            code = main(["--image", "unused.jpg", "--mock-text", "ignored"])
        self.assertEqual(code, 1)
        self.assertEqual(
            json.loads(stdout.getvalue()), {"error": "text_recognition_failed"}
        )

    def test_capture_saves_one_image_without_building_vision_client(self) -> None:
        ok, encoded = cv2.imencode(".jpg", np.full((20, 30, 3), 100, dtype=np.uint8))
        self.assertTrue(ok)
        stdout = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("text_recognition.cli.UsbLeftCamera") as camera,
            patch("text_recognition.cli.HttpVisionClient") as vision,
            contextlib.redirect_stdout(stdout),
        ):
            camera.return_value.capture.return_value = encoded.tobytes()
            code = main(
                [
                    "capture",
                    "--usb-port",
                    "COM_TEST",
                    "--output-dir",
                    directory,
                ]
            )
            payload = json.loads(stdout.getvalue())
            saved = Path(payload["path"])
            self.assertTrue(saved.is_file())
            self.assertEqual((payload["width"], payload["height"]), (30, 20))
        self.assertEqual(code, 0)
        vision.assert_not_called()

    def test_explicit_recognize_returns_json(self) -> None:
        stdout = io.StringIO()
        with (
            patch("text_recognition.cli.UsbLeftCamera") as camera,
            contextlib.redirect_stdout(stdout),
        ):
            camera.return_value.capture.return_value = b"jpeg"
            code = main(
                [
                    "recognize",
                    "--usb-port",
                    "COM_TEST",
                    "--mock-text",
                    "중앙도서관\n운영시간 09:00 - 22:00",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(stdout.getvalue()),
            {"text": "중앙도서관\n운영시간 09:00 - 22:00"},
        )

    def test_recognize_image_does_not_construct_usb_camera(self) -> None:
        stdout = io.StringIO()
        with (
            patch("text_recognition.cli.FileImageSource") as image_source,
            patch("text_recognition.cli.UsbLeftCamera") as usb_camera,
            contextlib.redirect_stdout(stdout),
        ):
            image_source.return_value.capture.return_value = b"jpeg"
            code = main(
                [
                    "recognize-image",
                    "--image",
                    "sample.jpg",
                    "--mock-text",
                    "안내문",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), {"text": "안내문"})
        usb_camera.assert_not_called()

    def test_preview_subcommand_runs_preview_controller(self) -> None:
        with (
            patch("text_recognition.cli.UsbPreviewCamera"),
            patch("text_recognition.cli.TextRecognitionPreview") as preview,
        ):
            code = main(
                ["preview", "--usb-port", "COM_TEST", "--mock-text", "테스트"]
            )
        self.assertEqual(code, 0)
        preview.return_value.run.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
