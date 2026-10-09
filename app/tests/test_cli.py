from __future__ import annotations

import os
import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from object_recognition.api import (
    DEFAULT_OPENAI_VISION_MODEL,
    HttpVisionClient,
    MockVisionClient,
)
from object_recognition.cli import (
    _build_vision_client,
    _parser,
    _run_usb,
    _usb_camera_id,
    main,
)
from object_recognition.models import RecognitionState


class CliTests(unittest.TestCase):
    def test_mock_is_default_and_model_can_be_overridden(self) -> None:
        parser = _parser()
        default_args = parser.parse_args(["--manual-pair", "front.jpg", "back.jpg"])
        self.assertFalse(default_args.live)
        self.assertEqual(default_args.model, DEFAULT_OPENAI_VISION_MODEL)
        self.assertIsInstance(_build_vision_client(default_args), MockVisionClient)

        live_args = parser.parse_args(
            ["--manual-pair", "front.jpg", "back.jpg", "--live", "--model", "custom-model"]
        )
        with patch.dict(os.environ, {"VIDEX_VISION_API_KEY": "test-key"}, clear=True):
            client = _build_vision_client(live_args)
        self.assertIsInstance(client, HttpVisionClient)
        self.assertEqual(client.model, "custom-model")

    def test_live_without_key_exits_before_reading_images_or_network(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("urllib.request.urlopen") as urlopen,
            self.assertLogs(level="ERROR") as logs,
        ):
            exit_code = main(["--manual-pair", "missing-front.jpg", "missing-back.jpg", "--live"])
        self.assertEqual(exit_code, 2)
        self.assertIn("VIDEX_VISION_API_KEY", "\n".join(logs.output))
        urlopen.assert_not_called()

    def test_usb_source_is_mutually_exclusive_and_camera_is_validated(self) -> None:
        parser = _parser()
        args = parser.parse_args(["--usb-port", "COM5", "--camera-id", "0"])
        self.assertEqual(args.usb_port, "COM5")
        self.assertEqual(_usb_camera_id(args), 0)
        default_args = parser.parse_args(["--usb-port", "COM5"])
        self.assertEqual(_usb_camera_id(default_args), 0)
        invalid_args = parser.parse_args(["--usb-port", "COM5", "--camera-id", "1"])
        with self.assertRaises(ValueError):
            _usb_camera_id(invalid_args)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--usb-port", "COM5", "--video", "sample.mp4"])

    def test_dual_camera_requires_usb_but_does_not_require_camera_id(self) -> None:
        args = _parser().parse_args(["--usb-port", "COM5", "--dual-camera"])
        self.assertTrue(args.dual_camera)
        with self.assertLogs(level="ERROR") as logs:
            exit_code = main(
                ["--manual-pair", "front.jpg", "back.jpg", "--dual-camera"]
            )
        self.assertEqual(exit_code, 2)
        self.assertIn("--usb-port", "\n".join(logs.output))

    def test_preview_option_and_stability_tuning_are_available(self) -> None:
        args = _parser().parse_args(
            [
                "--usb-port",
                "COM5",
                "--preview",
                "--motion-threshold",
                "0.04",
                "--motion-reset-threshold",
                "0.10",
                "--stable-ratio",
                "0.8",
            ]
        )
        self.assertTrue(args.preview)
        self.assertEqual(args.camera_id, "0")
        self.assertEqual(args.motion_threshold, 0.04)

    def test_product_recognition_has_no_runtime_resolution_option(self) -> None:
        parser = _parser()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--usb-port", "COM5", "--resolution", "2"])

    def test_usb_empty_poll_drives_recognizer_timeout_tick_and_closes(self) -> None:
        args = _parser().parse_args(["--usb-port", "COM5"])

        class FakeSource:
            stats = "fake-stats"
            closed = False

            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def __enter__(self) -> "FakeSource":
                return self

            def __exit__(self, *_args: object) -> None:
                self.closed = True

            def poll(self) -> list[object]:
                return []

        class FakeRecognizer:
            state = RecognitionState.WAIT_FRONT
            merged_path = None
            tick_calls = 0

            def tick(self) -> None:
                self.tick_calls += 1
                self.state = RecognitionState.COMPLETED

        recognizer = FakeRecognizer()
        source = FakeSource()
        with (
            patch("object_recognition.cli.UsbCameraSource", return_value=source),
            patch("object_recognition.cli._poll_usb_command", return_value=None),
        ):
            _run_usb(args, recognizer)  # type: ignore[arg-type]
        self.assertEqual(recognizer.tick_calls, 1)
        self.assertTrue(source.closed)

    def test_usb_capture_error_is_retried_without_hardware(self) -> None:
        args = _parser().parse_args(["--usb-port", "COM5", "--usb-retries", "1"])

        class FakeSource:
            stats = "fake-stats"

            def __enter__(self) -> "FakeSource":
                return self

            def __exit__(self, *_args: object) -> None:
                pass

            def poll(self) -> list[object]:
                raise AssertionError("retry should happen before serial polling")

        class FakeRecognizer:
            state = RecognitionState.ERROR
            merged_path = None
            retries = 0

            def retry_recognition(self) -> None:
                self.retries += 1
                self.state = RecognitionState.COMPLETED

        recognizer = FakeRecognizer()
        with (
            patch("object_recognition.cli.UsbCameraSource", return_value=FakeSource()),
            patch("object_recognition.cli._poll_usb_command", return_value=None),
        ):
            _run_usb(args, recognizer)  # type: ignore[arg-type]
        self.assertEqual(recognizer.retries, 1)


if __name__ == "__main__":
    unittest.main()
