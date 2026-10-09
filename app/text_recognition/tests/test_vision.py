from __future__ import annotations

import io
import json
import os
import socket
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from text_recognition.errors import (
    VisionApiError,
    VisionAuthenticationError,
    VisionResponseError,
    VisionTimeoutError,
)
from text_recognition.vision import (
    TEXT_RECOGNITION_PROMPT,
    HttpVisionClient,
    parse_text_response,
)


class VisionTests(unittest.TestCase):
    def test_payload_contains_one_image_prompt_and_strict_schema(self) -> None:
        client = HttpVisionClient()
        payload = client.build_payload(b"jpeg")
        messages = payload["messages"]
        self.assertEqual(messages[0]["content"], TEXT_RECOGNITION_PROMPT)
        content = messages[1]["content"]
        self.assertEqual(sum(item["type"] == "image_url" for item in content), 1)
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        response_format = payload["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        json_schema = response_format["json_schema"]
        self.assertTrue(json_schema["strict"])
        self.assertEqual(json_schema["schema"]["required"], ["text"])
        self.assertFalse(json_schema["schema"]["additionalProperties"])

    def test_multiline_sign_book_text_and_empty_text_are_preserved(self) -> None:
        sign = "서울책방\n중고서적 매입 및 판매\n영업시간 10:00 - 20:00"
        book = "인공지능은 인간의 학습과 추론 능력을 컴퓨터 시스템으로 구현하는 기술이다. " * 20
        for text in (sign, book, ""):
            with self.subTest(text=text[:20]):
                self.assertEqual(
                    parse_text_response(json.dumps({"text": text}, ensure_ascii=False)),
                    {"text": text},
                )

    def test_invalid_json_missing_wrong_or_extra_fields_raise(self) -> None:
        invalid = (
            "not-json",
            "[]",
            '{}',
            '{"text": null}',
            '{"text": "글씨", "extra": true}',
            '{"text": "첫째", "text": "둘째"}',
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(VisionResponseError):
                parse_text_response(raw)

    def test_missing_key_stops_before_network(self) -> None:
        client = HttpVisionClient()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("urllib.request.urlopen") as urlopen,
            self.assertRaises(VisionAuthenticationError),
        ):
            client.recognize(b"jpeg")
        urlopen.assert_not_called()

    def test_http_response_is_parsed_without_real_api(self) -> None:
        response = MagicMock()
        content = json.dumps(
            {"text": "3층 도서관\n2층 학생지원센터"}, ensure_ascii=False
        )
        response.__enter__.return_value.read.return_value = json.dumps(
            {"choices": [{"message": {"content": content}}]}, ensure_ascii=False
        ).encode("utf-8")
        client = HttpVisionClient()
        with (
            patch.dict(os.environ, {"VIDEX_VISION_API_KEY": "test-key"}, clear=True),
            patch("urllib.request.urlopen", return_value=response) as urlopen,
        ):
            result = client.recognize(b"jpeg")
        self.assertEqual(result, {"text": "3층 도서관\n2층 학생지원센터"})
        self.assertEqual(urlopen.call_count, 1)

    def test_auth_api_and_timeout_errors_remain_distinct(self) -> None:
        auth_error = urllib.error.HTTPError(
            "https://api.openai.com/v1/chat/completions",
            401,
            "Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b"provider detail"),
        )
        rate_error = urllib.error.HTTPError(
            "https://api.openai.com/v1/chat/completions",
            429,
            "Too Many Requests",
            hdrs=None,
            fp=None,
        )
        cases = (
            (auth_error, VisionAuthenticationError),
            (rate_error, VisionApiError),
            (urllib.error.URLError(socket.timeout()), VisionTimeoutError),
        )
        for error, expected in cases:
            with (
                self.subTest(error=error),
                patch.dict(os.environ, {"VIDEX_VISION_API_KEY": "secret"}, clear=True),
                patch("urllib.request.urlopen", side_effect=error),
                self.assertRaises(expected),
            ):
                HttpVisionClient().recognize(b"jpeg")


if __name__ == "__main__":
    unittest.main()
