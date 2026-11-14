"""SDK의 물리 요청별 사용량·실패를 기록하고 비용을 계산."""

import asyncio
import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from common.utils import now_iso
from config import API_MAX_RETRIES, API_TIMEOUT, BASE_DIR, PRICE_SNAPSHOT


def usage_cost(model: str | None, response: dict | None) -> dict:
    """실제 usage와 표준 처리 단가로 비용을 계산하고 미확보는 null로 보존."""
    missing = {"usage": None, "input_tokens": None, "cached_input_tokens": None,
               "output_tokens": None, "reasoning_tokens": None,
               "cost_usd": None, "cost_complete": False}
    if not response or not isinstance(response.get("usage"), dict):
        return missing
    usage = response["usage"]
    inputs = usage.get("input_tokens", usage.get("prompt_tokens"))
    outputs = usage.get("output_tokens", usage.get("completion_tokens"))
    is_embedding = bool(model and model.startswith("text-embedding-"))
    input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    cached = 0 if is_embedding else input_details.get("cached_tokens")
    outputs = 0 if is_embedding else outputs
    result = {"usage": usage, "input_tokens": inputs, "cached_input_tokens": cached,
              "output_tokens": outputs, "reasoning_tokens": output_details.get("reasoning_tokens"),
              "cost_usd": None, "cost_complete": False}
    rate = PRICE_SNAPSHOT["models"].get(model)
    tokens = (inputs, cached, outputs)
    valid = all(type(n) is int and n >= 0 for n in tokens)
    if (not rate or not valid or cached > inputs or (not is_embedding and inputs > 272000)
            or response.get("service_tier") not in (None, "default")):
        return result
    result["cost_usd"] = ((inputs - cached) * rate["input"]
                          + cached * (rate["cached_input"] or 0)
                          + outputs * (rate["output"] or 0)) / 1_000_000
    result["cost_complete"] = True
    return result


def aggregate_cost(calls: list[dict]) -> dict:
    """물리 호출 ID를 한 번만 합산하여 비용 완전성과 확인된 소계를 반환."""
    unique = {call["call_id"]: call for call in calls}
    complete = all(call["cost_complete"] for call in unique.values())
    known = sum(call["cost_usd"] for call in unique.values() if call["cost_usd"] is not None)
    return {"cost_usd": known if complete else None,
            "known_cost_usd": known, "cost_complete": complete}


class CallRecorder:
    """논리 호출과 SDK 재시도의 물리 요청을 연결하는 기록기."""

    def __init__(self) -> None:
        """호출별 문맥과 평가용 단일 요청 잠금을 준비."""
        self.calls: list[dict[str, Any]] = []
        self._context = ContextVar(f"api_scope_{id(self)}", default=None)
        self.lock = asyncio.Lock()

    @contextmanager
    def scope(self, purpose: str, question_id: str | None):
        """한 SDK 논리 호출의 용도와 문항을 재시도 전체에 연결."""
        context = {"logical_call_id": uuid4().hex,
                   "purpose": purpose, "question_id": question_id}
        token = self._context.set(context)
        try:
            yield context
        finally:
            self._context.reset(token)

    def start(self, request: httpx.Request) -> tuple[dict, float]:
        """인증 헤더 없이 실제 요청 본문과 재시도 번호를 기록."""
        context = self._context.get() or {"logical_call_id": uuid4().hex,
                                          "purpose": "unspecified", "question_id": None}
        raw_request = request.content.decode("utf-8")
        request_body = json.loads(raw_request)
        # SDK retry header는 인증 정보가 아니며 추가 논리 호출과 구분하는 데 사용합니다.
        attempt = int(request.headers.get("x-stainless-retry-count", "0")) + 1
        entry = {"call_id": uuid4().hex, **context, "attempt": attempt,
                 "endpoint": request.url.path, "model": request_body.get("model"),
                 "request_body": request_body, "raw_request": raw_request,
                 "response_body": None, "raw_response": None, "request_id": None,
                 "status_code": None, "service_tier": None, "started_at": now_iso(),
                 "finished_at": None, "elapsed_ms": None, "error": None,
                 **usage_cost(None, None)}
        self.calls.append(entry)
        return entry, time.perf_counter()

    def finish(self, entry: dict, start: float, response: httpx.Response | None,
               error: BaseException | None = None) -> None:
        """실패 응답도 원문·실제 usage를 보존하고 경과 시간을 확정."""
        entry["finished_at"] = now_iso()
        entry["elapsed_ms"] = (time.perf_counter() - start) * 1000
        if error is not None:
            entry["error"] = {"code": type(error).__name__, "message": "HTTP 요청 또는 응답 수신 실패"}
        if response is None:
            return
        entry["status_code"] = response.status_code
        entry["request_id"] = response.headers.get("x-request-id")
        try:
            content = response.content
        except httpx.ResponseNotRead:
            # 본문 수신 중 끊겼어도 이미 받은 상태·요청 ID를 버리지 않습니다.
            content = None
        if content is not None:
            entry["raw_response"] = content.decode("utf-8", errors="replace")
        try:
            body = json.loads(content) if content is not None else None
        except (ValueError, UnicodeDecodeError):
            body = None
        if isinstance(body, dict):
            entry["response_body"] = body
            entry["service_tier"] = body.get("service_tier")
            entry.update(usage_cost(entry["model"], body))
        if response.is_error and error is None:
            entry["error"] = {"code": f"http_{response.status_code}", "message": "API 오류 응답"}


class RecordingTransport(httpx.BaseTransport):
    """동기 SDK의 재시도까지 포함해 HTTP 송수신을 수집."""

    def __init__(self, recorder: CallRecorder, inner: httpx.BaseTransport | None = None):
        """실제 또는 테스트용 전송기에 기록기를 연결."""
        self.recorder, self.inner = recorder, inner or httpx.HTTPTransport(retries=0)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """응답을 끝까지 읽어 기록한 뒤 SDK에 전달."""
        entry, start = self.recorder.start(request)
        response = None
        try:
            response = self.inner.handle_request(request)
            response.read()
        except BaseException as error:
            self.recorder.finish(entry, start, response, error)
            if response is not None:
                response.close()
            raise
        self.recorder.finish(entry, start, response)
        return response

    def close(self) -> None:
        """하위 전송 자원을 닫음."""
        self.inner.close()


class AsyncRecordingTransport(httpx.AsyncBaseTransport):
    """지표 내부 병렬 작업도 실제 HTTP 요청은 한 번에 하나로 제한."""

    def __init__(self, recorder: CallRecorder, inner: httpx.AsyncBaseTransport | None = None):
        """공유 기록기 잠금과 비동기 전송기를 준비."""
        self.recorder, self.inner = recorder, inner or httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """단일 잠금 안에서 물리 요청·응답과 실패를 기록."""
        async with self.recorder.lock:
            entry, start = self.recorder.start(request)
            response = None
            try:
                response = await self.inner.handle_async_request(request)
                await response.aread()
            except BaseException as error:
                self.recorder.finish(entry, start, response, error)
                if response is not None:
                    await response.aclose()
                raise
            self.recorder.finish(entry, start, response)
            return response

    async def aclose(self) -> None:
        """비동기 전송 자원을 닫음."""
        await self.inner.aclose()


def load_credentials() -> None:
    """사용자가 유료 실행 명령을 호출했을 때만 로컬 키를 로드."""
    load_dotenv(BASE_DIR / ".env", override=False)
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "replace_with_your_key":
        raise ValueError("OPENAI_API_KEY를 환경 변수 또는 week-03/DChanHong/.env에 설정하세요.")


def make_client(recorder: CallRecorder) -> OpenAI:
    """확정한 timeout·재시도로 동기 생성 클라이언트를 구성."""
    load_credentials()
    return OpenAI(timeout=API_TIMEOUT, max_retries=API_MAX_RETRIES,
                  base_url="https://api.openai.com/v1",
                  http_client=httpx.Client(transport=RecordingTransport(recorder)))


def make_async_client(recorder: CallRecorder) -> AsyncOpenAI:
    """확정한 timeout·재시도로 단일 동시 요청 평가 클라이언트를 구성."""
    load_credentials()
    return AsyncOpenAI(timeout=API_TIMEOUT, max_retries=API_MAX_RETRIES,
                       base_url="https://api.openai.com/v1",
                       http_client=httpx.AsyncClient(transport=AsyncRecordingTransport(recorder)))
