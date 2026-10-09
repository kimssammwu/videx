"""Replaceable Vision API clients with opt-in OpenAI network access."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


PRODUCT_PROMPT = (
    "이미지에 있는 제품의 정확한 제품명만 출력하세요. 브랜드와 용량이 제품명에 포함되어 "
    "있다면 유지하세요. 설명이나 다른 문장은 출력하지 마세요. 식별할 수 없다면 "
    "'인식 실패'만 출력하세요."
)
OPENAI_CHAT_COMPLETIONS_ENDPOINT = "https://api.openai.com/v1/chat/completions"
DEFAULT_OPENAI_VISION_MODEL = "gpt-4.1-mini"


class VisionApiError(RuntimeError):
    pass


class VisionClient(Protocol):
    def recognize(self, jpeg_bytes: bytes) -> str:
        """Return one validated product-name string or '인식 실패'."""


def parse_product_name(raw: str) -> str:
    """Strictly accept a single short plain-text product name."""

    if not isinstance(raw, str):
        raise VisionApiError("Vision API 응답이 문자열이 아닙니다.")
    value = raw.strip().strip("\"'").strip()
    if value == "인식 실패":
        return value
    if not value or len(value) > 100:
        raise VisionApiError("제품명이 비어 있거나 너무 깁니다.")
    if "\n" in value or "\r" in value:
        raise VisionApiError("제품명 외의 여러 줄 응답을 받았습니다.")
    if value.startswith(("{", "[", "```", "#", "- ")):
        raise VisionApiError("제품명 형식이 아닌 응답을 받았습니다.")
    if re.search(r"(?i)^(product_name|제품명)\s*[:=]", value):
        raise VisionApiError("설명 필드가 포함된 응답을 받았습니다.")
    return value


@dataclass
class MockVisionClient:
    product_name: str = "테스트 제품 1L"
    calls: int = 0

    def recognize(self, jpeg_bytes: bytes) -> str:
        if not jpeg_bytes:
            raise VisionApiError("빈 이미지입니다.")
        self.calls += 1
        return parse_product_name(self.product_name)


class CachingVisionClient:
    """Prevent duplicate billable calls for identical merged JPEG payloads."""

    def __init__(self, inner: VisionClient) -> None:
        self.inner = inner
        self._cache: dict[str, str] = {}
        self._attempted: set[str] = set()

    def recognize(self, jpeg_bytes: bytes) -> str:
        digest = hashlib.sha256(jpeg_bytes).hexdigest()
        if digest in self._cache:
            return self._cache[digest]
        if digest in self._attempted:
            raise VisionApiError("동일한 이미지에 대한 API 호출은 이미 시도되었습니다.")
        self._attempted.add(digest)
        self._cache[digest] = parse_product_name(self.inner.recognize(jpeg_bytes))
        return self._cache[digest]


class HttpVisionClient:
    """Minimal OpenAI Chat Completions transport for one merged JPEG image."""

    def __init__(
        self,
        endpoint: str = OPENAI_CHAT_COMPLETIONS_ENDPOINT,
        model: str = DEFAULT_OPENAI_VISION_MODEL,
        api_key_env: str = "VIDEX_VISION_API_KEY",
        timeout: float = 20.0,
        max_output_tokens: int = 48,
    ) -> None:
        if not endpoint.lower().startswith("https://"):
            raise ValueError("Vision API endpoint must use HTTPS")
        self.endpoint = endpoint
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

    def build_payload(self, jpeg_bytes: bytes) -> dict[str, object]:
        if not jpeg_bytes:
            raise VisionApiError("빈 이미지입니다.")
        encoded = base64.b64encode(jpeg_bytes).decode("ascii")
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PRODUCT_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            "max_completion_tokens": self.max_output_tokens,
            "n": 1,
            "temperature": 0,
        }

    def validate_configuration(self) -> None:
        """Fail before image processing or HTTP access when the key is missing."""

        api_key = os.environ.get(self.api_key_env)
        if not api_key or not api_key.strip():
            raise VisionApiError(f"환경변수 {self.api_key_env}가 설정되지 않았습니다.")

    def recognize(self, jpeg_bytes: bytes) -> str:
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
            raise VisionApiError(self._safe_http_error(exc.code)) from None
        except (urllib.error.URLError, TimeoutError):
            raise VisionApiError("OpenAI API 네트워크 요청에 실패했습니다.") from None
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError, TypeError):
            raise VisionApiError("OpenAI API 응답 형식을 처리할 수 없습니다.") from None
        return parse_product_name(raw)

    @staticmethod
    def _safe_http_error(status: int) -> str:
        messages = {
            400: "요청 형식이 올바르지 않습니다.",
            401: "API 키 인증에 실패했습니다.",
            403: "API 접근 권한이 없습니다.",
            404: "API 엔드포인트 또는 모델을 찾을 수 없습니다.",
            413: "전송 이미지가 너무 큽니다.",
            429: "요청 한도 또는 사용량 한도에 도달했습니다.",
        }
        if status >= 500:
            detail = "OpenAI 서버에서 요청을 처리하지 못했습니다."
        else:
            detail = messages.get(status, "요청이 거부되었습니다.")
        return f"OpenAI API 오류 (HTTP {status}): {detail}"
