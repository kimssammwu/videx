from __future__ import annotations

import io
import os
import urllib.error
import unittest
from unittest.mock import MagicMock, patch

from object_recognition.api import (
    DEFAULT_OPENAI_VISION_MODEL,
    OPENAI_CHAT_COMPLETIONS_ENDPOINT,
    PRODUCT_PROMPT,
    CachingVisionClient,
    HttpVisionClient,
    MockVisionClient,
    VisionApiError,
    parse_product_name,
)


class ApiTests(unittest.TestCase):
    def test_product_name_validation(self) -> None:
        self.assertEqual(parse_product_name('  "서울우유 나100% 1L"  '), "서울우유 나100% 1L")
        self.assertEqual(parse_product_name("인식 실패"), "인식 실패")
        for invalid in ("", "제품명: 우유", "첫 줄\n둘째 줄", '{"product_name":"우유"}'):
            with self.subTest(invalid=invalid), self.assertRaises(VisionApiError):
                parse_product_name(invalid)

    def test_cache_prevents_duplicate_calls(self) -> None:
        mock = MockVisionClient("서울우유 나100% 1L")
        client = CachingVisionClient(mock)
        self.assertEqual(client.recognize(b"jpeg"), "서울우유 나100% 1L")
        self.assertEqual(client.recognize(b"jpeg"), "서울우유 나100% 1L")
        self.assertEqual(mock.calls, 1)

    def test_http_payload_is_single_image_and_short_output(self) -> None:
        client = HttpVisionClient()
        payload = client.build_payload(b"jpeg")
        self.assertEqual(client.endpoint, OPENAI_CHAT_COMPLETIONS_ENDPOINT)
        self.assertEqual(payload["model"], DEFAULT_OPENAI_VISION_MODEL)
        self.assertEqual(payload["max_completion_tokens"], 48)
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(payload["n"], 1)
        content = payload["messages"][0]["content"]  # type: ignore[index]
        self.assertEqual(content[0]["text"], PRODUCT_PROMPT)
        self.assertEqual(sum(part["type"] == "image_url" for part in content), 1)
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(content[1]["image_url"]["detail"], "high")

    def test_missing_key_stops_before_http_request(self) -> None:
        client = HttpVisionClient()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("urllib.request.urlopen") as urlopen,
            self.assertRaisesRegex(VisionApiError, "VIDEX_VISION_API_KEY"),
        ):
            client.recognize(b"jpeg")
        urlopen.assert_not_called()

    def test_http_error_exposes_status_but_not_key_or_response(self) -> None:
        secret = "test-secret-that-must-not-appear"
        response_body = b'{"error":{"message":"sensitive provider detail"}}'
        error = urllib.error.HTTPError(
            OPENAI_CHAT_COMPLETIONS_ENDPOINT,
            401,
            "Unauthorized",
            hdrs=None,
            fp=io.BytesIO(response_body),
        )
        client = HttpVisionClient()
        with (
            patch.dict(os.environ, {"VIDEX_VISION_API_KEY": secret}, clear=True),
            patch("urllib.request.urlopen", side_effect=error),
            self.assertRaises(VisionApiError) as context,
        ):
            client.recognize(b"jpeg")
        message = str(context.exception)
        self.assertIn("HTTP 401", message)
        self.assertNotIn(secret, message)
        self.assertNotIn("sensitive provider detail", message)

    def test_failed_image_is_not_sent_twice(self) -> None:
        error = urllib.error.HTTPError(
            OPENAI_CHAT_COMPLETIONS_ENDPOINT,
            429,
            "Too Many Requests",
            hdrs=None,
            fp=None,
        )
        client = CachingVisionClient(HttpVisionClient())
        with (
            patch.dict(os.environ, {"VIDEX_VISION_API_KEY": "unit-test-placeholder"}, clear=True),
            patch("urllib.request.urlopen", side_effect=error) as urlopen,
        ):
            with self.assertRaisesRegex(VisionApiError, "HTTP 429"):
                client.recognize(b"same-merged-jpeg")
            with self.assertRaisesRegex(VisionApiError, "이미 시도"):
                client.recognize(b"same-merged-jpeg")
        self.assertEqual(urlopen.call_count, 1)

    def test_openai_response_is_parsed_without_real_network_access(self) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = (
            b'{"choices":[{"message":{"content":"Seoul Milk 1L"}}]}'
        )
        client = CachingVisionClient(HttpVisionClient())
        with (
            patch.dict(os.environ, {"VIDEX_VISION_API_KEY": "test-key"}, clear=True),
            patch("urllib.request.urlopen", return_value=response) as urlopen,
        ):
            self.assertEqual(client.recognize(b"merged-jpeg"), "Seoul Milk 1L")
            self.assertEqual(client.recognize(b"merged-jpeg"), "Seoul Milk 1L")
        self.assertEqual(urlopen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
