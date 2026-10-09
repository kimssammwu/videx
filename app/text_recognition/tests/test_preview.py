from __future__ import annotations

import tempfile
import threading
import time
import unittest
from collections.abc import Callable
from pathlib import Path

import numpy as np

from text_recognition.camera import PreviewFrame
from text_recognition.preview import CaptureStore, TextRecognitionPreview
from text_recognition.vision import MockVisionClient


class FakeSource:
    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self.poll_calls = 0
        self.image = np.full((36, 48, 3), 180, dtype=np.uint8)

    def open(self) -> "FakeSource":
        self.opened = True
        return self

    def poll(self) -> list[PreviewFrame]:
        self.poll_calls += 1
        return [PreviewFrame(self.image.copy(), self.poll_calls, time.monotonic())]

    def close(self) -> None:
        self.closed = True


class FakeUI:
    def __init__(
        self,
        commands: list[str | None],
        *,
        on_quit: Callable[[], None] | None = None,
    ) -> None:
        self.commands = list(commands)
        self.on_quit = on_quit
        self.opened = False
        self.closed = False
        self.frames_seen = 0

    def open(self) -> None:
        self.opened = True

    def show(self, frame: PreviewFrame | None, _status: str) -> str | None:
        if frame is not None:
            self.frames_seen += 1
        time.sleep(0.01)
        command = self.commands.pop(0) if self.commands else "quit"
        if command == "quit" and self.on_quit is not None:
            self.on_quit()
        return command

    def close(self) -> None:
        self.closed = True


class BlockingVision:
    def __init__(self) -> None:
        self.calls = 0
        self.release = threading.Event()

    def recognize(self, _jpeg_bytes: bytes) -> dict[str, str]:
        self.calls += 1
        if not self.release.wait(timeout=2):
            raise AssertionError("test did not release Vision worker")
        return {"text": "완료"}


class FailingVision:
    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, _jpeg_bytes: bytes) -> dict[str, str]:
        self.calls += 1
        raise RuntimeError("provider unavailable")


class PreviewTests(unittest.TestCase):
    def _run(
        self,
        directory: str,
        commands: list[str | None],
        vision: object,
        *,
        on_quit: Callable[[], None] | None = None,
    ) -> tuple[FakeSource, FakeUI, list[dict[str, str]], list[str]]:
        source = FakeSource()
        ui = FakeUI(commands, on_quit=on_quit)
        results: list[dict[str, str]] = []
        events: list[str] = []
        preview = TextRecognitionPreview(
            source,
            vision,
            CaptureStore(directory),
            ui=ui,
            result_writer=results.append,
            event_writer=events.append,
        )
        preview.run()
        return source, ui, results, events

    def test_preview_does_not_call_api_automatically_and_q_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            vision = MockVisionClient("사용되면 안 됨")
            source, ui, results, _events = self._run(
                directory, [None, "quit"], vision
            )
            self.assertEqual(list(Path(directory).glob("*.jpg")), [])
        self.assertEqual(vision.calls, 0)
        self.assertEqual(results, [])
        self.assertTrue(source.opened and source.closed)
        self.assertTrue(ui.opened and ui.closed)
        self.assertGreater(ui.frames_seen, 0)

    def test_space_saves_and_sends_the_exact_same_jpeg_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            vision = MockVisionClient("서울책방\n영업시간 10:00 - 20:00")
            _source, _ui, results, _events = self._run(
                directory, ["capture", None, None, "quit"], vision
            )
            files = list(Path(directory).glob("*.jpg"))
            self.assertEqual(len(files), 1)
            self.assertEqual(vision.last_image, files[0].read_bytes())
        self.assertEqual(vision.calls, 1)
        self.assertEqual(
            results, [{"text": "서울책방\n영업시간 10:00 - 20:00"}]
        )

    def test_s_saves_without_api_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            vision = MockVisionClient("사용되면 안 됨")
            self._run(directory, ["save", "quit"], vision)
            files = list(Path(directory).glob("*.jpg"))
            self.assertEqual(len(files), 1)
        self.assertEqual(vision.calls, 0)

    def test_repeated_space_is_rejected_while_api_is_busy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            vision = BlockingVision()
            _source, _ui, _results, events = self._run(
                directory,
                ["capture", "capture", "quit"],
                vision,
                on_quit=vision.release.set,
            )
            self.assertEqual(len(list(Path(directory).glob("*.jpg"))), 1)
        self.assertEqual(vision.calls, 1)
        self.assertTrue(any("처리 중" in event for event in events))

    def test_api_failure_does_not_prevent_later_save(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            vision = FailingVision()
            source, ui, results, events = self._run(
                directory, ["capture", None, "save", "quit"], vision
            )
            self.assertEqual(len(list(Path(directory).glob("*.jpg"))), 2)
        self.assertEqual(vision.calls, 1)
        self.assertEqual(results, [])
        self.assertTrue(any("provider unavailable" in event for event in events))
        self.assertTrue(source.closed and ui.closed)


if __name__ == "__main__":
    unittest.main()
