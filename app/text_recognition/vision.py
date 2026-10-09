"""OpenAI Vision transport and strict text-result validation."""

from __future__ import annotations

import base64
import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .config import (
    DEFAULT_VISION_ENDPOINT,
    DEFAULT_VISION_MODEL,
    VISION_API_KEY_ENV,
)
from .errors import (
    VisionApiError,
    VisionAuthenticationError,
    VisionResponseError,
    VisionTimeoutError,
)


TEXT_RECOGNITION_PROMPT = """당신은 시각장애인을 위한 범용 이미지 텍스트 인식 시스템입니다.

입력된 이미지에 실제로 보이는 글씨를 가능한 한 정확하게 추출하세요.

이미지에는 간판, 책, 표지판, 문서, 메뉴판, 포스터, 팸플릿 등 다양한 글씨가 포함될 수 있습니다.

규칙:

1. 이미지에 실제로 보이는 한글, 영어, 숫자, 기호를 최대한 정확하게 인식하세요.
2. 이미지에 보이지 않는 단어나 문장을 추측하여 생성하지 마세요.
3. 원문을 요약하거나 번역하지 마세요.
4. 여러 줄의 글씨는 자연스러운 읽기 순서대로 정리하세요.
5. 책이나 문서의 경우 문장과 문단 순서를 최대한 유지하세요.
6. 간판과 안내판의 경우 명확하게 보이는 텍스트를 자연스러운 순서로 추출하세요.
7. 메뉴판과 목록은 항목별 줄바꿈을 유지하세요.
8. 여러 단으로 나뉜 문서는 가능한 한 단별 읽기 순서를 유지하세요.
9. 불분명한 글자를 임의로 완성하지 마세요.
10. 글씨의 의미나 이미지 속 배경을 설명하지 마세요.
11. 글씨가 없다면 빈 문자열을 반환하세요.
12. 반드시 유효한 JSON 객체 하나만 반환하세요.
13. JSON 외에는 어떤 설명도 추가하지 마세요.

응답 형식:

{
  "text": "이미지에서 인식한 글씨"
}"""

TEXT_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


class VisionClient(Protocol):
    def recognize(self, jpeg_bytes: bytes) -> dict[str, str]:
        """Recognize visible text and return exactly ``{"text": str}``."""


def parse_text_response(raw: str) -> dict[str, str]:
    """Strictly validate the provider's JSON text without silent fallback."""

    if not isinstance(raw, str):
        raise VisionResponseError("Vision API message content가 문자열이 아닙니다.")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise VisionResponseError(
                    "Vision API 응답에 중복된 JSON 필드가 있습니다."
                )
            value[key] = item
        return value

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise VisionResponseError("Vision API 응답이 유효한 JSON이 아닙니다.") from exc
    if not isinstance(value, dict):
        raise VisionResponseError("Vision API 응답은 JSON 객체여야 합니다.")
    if set(value) != {"text"}:
        raise VisionResponseError("Vision API 응답에는 text 필드 하나만 있어야 합니다.")
    if not isinstance(value["text"], str):
        raise VisionResponseError("Vision API 응답의 text 값은 문자열이어야 합니다.")
    return {"text": value["text"]}


@dataclass
class MockVisionClient:
    """Deterministic, network-free client for tests and CLI smoke checks."""

    text: str = "테스트 글씨"
    calls: int = 0
    last_image: bytes | None = None

    def recognize(self, jpeg_bytes: bytes) -> dict[str, str]:
        if not jpeg_bytes:
            raise VisionApiError("빈 이미지입니다.")
        self.calls += 1
        self.last_image = jpeg_bytes
        return {"text": self.text}


class HttpVisionClient:
    """Minimal OpenAI Chat Completions client with Structured Outputs."""

    def __init__(
        self,
        *,
        endpoint: str = DEFAULT_VISION_ENDPOINT,
        model: str = DEFAULT_VISION_MODEL,
        api_key_env: str = VISION_API_KEY_ENV,
        timeout: float = 30.0,
        max_output_tokens: int = 8192,
    ) -> None:
        if not endpoint.lower().startswith("https://"):
            raise ValueError("Vision API endpoint는 HTTPS를 사용해야 합니다.")
        if timeout <= 0 or max_output_tokens <= 0:
            raise ValueError("timeout과 max_output_tokens는 0보다 커야 합니다.")
        self.endpoint = endpoint
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

    def validate_configuration(self) -> None:
        api_key = os.environ.get(self.api_key_env)
        if not api_key or not api_key.strip():
            raise VisionAuthenticationError(
                f"환경변수 {self.api_key_env}가 설정되지 않았습니다."
            )

    def build_payload(self, jpeg_bytes: bytes) -> dict[str, object]:
        if not jpeg_bytes:
            raise VisionApiError("빈 이미지입니다.")
        encoded = base64.b64encode(jpeg_bytes).decode("ascii")
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": TEXT_RECOGNITION_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "이 이미지에 실제로 보이는 글씨를 JSON으로 추출하세요.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "videx_text_recognition",
                    "strict": True,
                    "schema": TEXT_RESPONSE_SCHEMA,
                },
            },
            "max_completion_tokens": self.max_output_tokens,
            "n": 1,
            "temperature": 0,
        }

    def recognize(self, jpeg_bytes: bytes) -> dict[str, str]:
        self.validate_configuration()
        api_key = os.environ[self.api_key_env].strip()
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(self.build_payload(jpeg_bytes)).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            raw = payload["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            self._raise_http_error(exc.code)
        except (TimeoutError, socket.timeout) as exc:
            raise VisionTimeoutError("Vision API 요청 시간이 초과되었습니다.") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise VisionTimeoutError("Vision API 요청 시간이 초과되었습니다.") from exc
            raise VisionApiError("Vision API 네트워크 요청에 실패했습니다.") from exc
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError, TypeError) as exc:
            raise VisionResponseError(
                "Vision API 응답 구조를 처리할 수 없습니다."
            ) from exc
        return parse_text_response(raw)

    @staticmethod
    def _raise_http_error(status: int) -> None:
        if status in (401, 403):
            raise VisionAuthenticationError(
                f"Vision API 인증 또는 권한 확인에 실패했습니다 (HTTP {status})."
            )
        if status == 408:
            raise VisionTimeoutError("Vision API 요청 시간이 초과되었습니다 (HTTP 408).")
        messages = {
            400: "요청 형식이 올바르지 않습니다.",
            404: "API endpoint 또는 model을 찾을 수 없습니다.",
            413: "전송 이미지가 너무 큽니다.",
            429: "요청 한도 또는 사용량 한도에 도달했습니다.",
        }
        detail = (
            "OpenAI 서버에서 요청을 처리하지 못했습니다."
            if status >= 500
            else messages.get(status, "요청이 거부되었습니다.")
        )
        raise VisionApiError(f"Vision API 오류 (HTTP {status}): {detail}")
