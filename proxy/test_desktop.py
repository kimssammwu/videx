"""Desktop configuration contract: native local defaults and explicit mocks."""
from pathlib import Path
import tempfile
import unittest

from proxy.desktop import clients, parser, saved_mode
from proxy.runtime import CameraSource
from app.object_recognition.api import HttpVisionClient as ObjectAPI
from app.text_recognition.vision import HttpVisionClient as TextAPI


class DesktopTests(unittest.TestCase):
    def test_defaults_use_local_speech_and_real_api_clients(self):
        args = parser().parse_args([])
        self.assertFalse(args.mute)
        self.assertFalse(args.mock)
        self.assertFalse(args.demo)
        text, objects = clients(args, False)
        self.assertIsInstance(text, TextAPI)
        self.assertIsInstance(objects, ObjectAPI)

    def test_both_recognition_modes_request_640_resolution_without_opening_new_camera(self):
        source = CameraSource(port="COM_TEST")
        source.set_mode(0)
        self.assertEqual(source.control["resolution"], 0)
        source.set_mode(1)
        self.assertEqual(source.control["resolution"], 1)
        source.set_mode(2)
        self.assertEqual(source.control["resolution"], 1)
        self.assertIsNone(source.thread)

    def test_last_mode_restores_and_bad_settings_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.assertEqual(saved_mode(path), 0)
            for contents, expected in [('{"mode":2}', 2), ('{"mode":99}', 0),
                                       ('{"mode":true}', 0), ('broken', 0), ('{}', 0)]:
                (path / "settings.json").write_text(contents, encoding="utf-8")
                self.assertEqual(saved_mode(path), expected)


if __name__ == "__main__":
    unittest.main()
