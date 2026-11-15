"""로컬 모의 전송기로 SDK 재시도·비용·비밀 정보 제외를 확인한다."""

import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
from openai import APITimeoutError, AsyncOpenAI, OpenAI

from common.api import (
    AsyncRecordingTransport,
    CallRecorder,
    RecordingTransport,
    aggregate_cost,
    usage_cost,
)
from config import EVALUATION_MODEL


def completion_body():
    """실제 토큰 필드를 가진 오프라인 Chat Completions 응답을 생성."""
    return {
        "id": "mock-completion", "object": "chat.completion", "created": 1,
        "model": EVALUATION_MODEL, "service_tier": "default",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "합성 답변"},
                     "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200,
            "prompt_tokens_details": {"cached_tokens": 100},
            "completion_tokens_details": {"reasoning_tokens": 70},
        },
    }


class BrokenStream(httpx.SyncByteStream):
    """헤더 수신 후 본문이 끊기는 네트워크 오류를 모의."""

    def __iter__(self):
        """불완전한 본문 다음에 실제 네트워크 없이 읽기 오류를 발생."""
        yield b'{"id":"partial"'
        raise httpx.ReadTimeout("offline read timeout")


class CostTests(unittest.TestCase):
    """usage 원본·캐시 할인·부분 비용 합계의 의미를 검증."""

    def test_cached_discount_and_reasoning_are_not_double_counted(self):
        """캐시 입력은 할인하고 출력에 포함된 추론은 다시 더하지 않음."""
        result = usage_cost(EVALUATION_MODEL, completion_body())
        self.assertTrue(result["cost_complete"])
        self.assertEqual(result["reasoning_tokens"], 70)
        self.assertAlmostEqual(result["cost_usd"], (900 * 2.5 + 100 * 0.25 + 200 * 15) / 1_000_000)

    def test_missing_usage_and_cache_details_are_unknown(self):
        """미확보 사용량과 캐시 세부 정보를 임의로 0으로 채우지 않음."""
        unknown = usage_cost(EVALUATION_MODEL, {"error": {"message": "합성 오류"}})
        self.assertIsNone(unknown["cost_usd"])
        self.assertFalse(unknown["cost_complete"])
        body = completion_body()
        del body["usage"]["prompt_tokens_details"]
        missing_cache = usage_cost(EVALUATION_MODEL, body)
        self.assertIsNone(missing_cache["cached_input_tokens"])
        self.assertFalse(missing_cache["cost_complete"])
        self.assertEqual(missing_cache["usage"], body["usage"])

    def test_embedding_batch_has_no_llm_long_context_surcharge(self):
        """임베딩 입력에는 LLM 장문 가격 제한을 적용하지 않음."""
        result = usage_cost("text-embedding-3-small", {"usage": {"prompt_tokens": 280000}})
        self.assertTrue(result["cost_complete"])
        self.assertAlmostEqual(result["cost_usd"], 280000 * 0.02 / 1_000_000)

    def test_unknown_cost_preserves_known_subtotal_and_deduplicates_ids(self):
        """같은 물리 요청은 한 번만 합산하고 미확보 비용은 따로 유지."""
        calls = [
            {"call_id": "known", "cost_usd": 0.1, "cost_complete": True},
            {"call_id": "unknown", "cost_usd": None, "cost_complete": False},
        ]
        total = aggregate_cost(calls + calls)
        self.assertEqual(total, {"cost_usd": None, "known_cost_usd": 0.1, "cost_complete": False})
        # 복사된 과거 RAG 호출과 이번 평가 호출의 합계는 각각 따로 유지한다.
        evaluation = aggregate_cost([{"call_id": "new_eval", "cost_usd": 0.02, "cost_complete": True}])
        self.assertEqual(evaluation["cost_usd"], 0.02)


class TransportTests(unittest.TestCase):
    """SDK에 실제 모의 HTTP 전송기를 연결하여 물리 시도를 확인."""

    def test_sdk_retries_keep_attempts_and_one_logical_id_without_auth(self):
        """SDK 재시도 2회는 물리 기록 3개이며 인증 헤더는 저장하지 않음."""
        recorder = CallRecorder()
        seen_headers = []

        def handler(request):
            """첫 두 요청은 실패시키고 세 번째는 사용량과 함께 반환."""
            seen_headers.append(dict(request.headers))
            number = len(seen_headers)
            if number < 3:
                return httpx.Response(429, json={"error": {"message": "offline retry", "type": "rate_limit_error"}},
                                      headers={"x-request-id": f"mock-{number}"})
            return httpx.Response(200, json=completion_body(), headers={"x-request-id": "mock-3"})

        transport = RecordingTransport(recorder, httpx.MockTransport(handler))
        with OpenAI(api_key="offline-dummy-secret", base_url="https://offline.invalid/v1", max_retries=2,
                    http_client=httpx.Client(transport=transport)) as client:
            with patch.object(client, "_calculate_retry_timeout", return_value=0):
                with recorder.scope("test_judge", "Q01"):
                    result = client.chat.completions.create(
                        model=EVALUATION_MODEL, messages=[{"role": "user", "content": "합성 질문"}],
                        service_tier="default", store=False,
                    )

        self.assertEqual(result.choices[0].message.content, "합성 답변")
        self.assertEqual([call["attempt"] for call in recorder.calls], [1, 2, 3])
        self.assertEqual([call["status_code"] for call in recorder.calls], [429, 429, 200])
        self.assertEqual(len({call["logical_call_id"] for call in recorder.calls}), 1)
        self.assertEqual(len({call["call_id"] for call in recorder.calls}), 3)
        self.assertTrue(all(call["question_id"] == "Q01" for call in recorder.calls))
        self.assertTrue(all(call["elapsed_ms"] >= 0 for call in recorder.calls))
        self.assertIn("authorization", seen_headers[0])
        saved = json.dumps(recorder.calls, ensure_ascii=False)
        self.assertNotIn("offline-dummy-secret", saved)
        self.assertNotIn("authorization", saved.lower())
        total = aggregate_cost(recorder.calls)
        self.assertFalse(total["cost_complete"])
        self.assertIsNone(total["cost_usd"])
        self.assertEqual(total["known_cost_usd"], recorder.calls[-1]["cost_usd"])

    def test_interrupted_response_preserves_received_headers(self):
        """본문 수신 실패에도 이미 받은 상태·요청 ID는 유지."""
        recorder = CallRecorder()

        def handler(request):
            """외부 연결 없이 부분 본문이 있는 모의 응답을 반환."""
            return httpx.Response(200, stream=BrokenStream(), headers={"x-request-id": "received-before-timeout"})

        transport = RecordingTransport(recorder, httpx.MockTransport(handler))
        with OpenAI(api_key="offline-dummy-secret", base_url="https://offline.invalid/v1", max_retries=0,
                    http_client=httpx.Client(transport=transport)) as client:
            with recorder.scope("test_timeout", "Q02"):
                with self.assertRaises(APITimeoutError):
                    client.chat.completions.create(model=EVALUATION_MODEL, messages=[{"role": "user", "content": "합성 질문"}])
        self.assertEqual(len(recorder.calls), 1)
        call = recorder.calls[0]
        self.assertEqual(call["status_code"], 200)
        self.assertEqual(call["request_id"], "received-before-timeout")
        self.assertEqual(call["error"]["code"], "ReadTimeout")
        self.assertIsNone(call["raw_response"])
        self.assertIsNone(call["cost_usd"])
        self.assertFalse(call["cost_complete"])


class AsyncTransportTests(unittest.IsolatedAsyncioTestCase):
    """지표가 동시에 호출해도 물리 요청을 직렬화하는지 확인."""

    async def test_shared_recorder_limits_concurrency_to_one(self):
        """여러 논리 호출의 문항 문맥을 보존하면서 HTTP 동시성은 1."""
        recorder = CallRecorder()
        active = 0
        peak = 0

        async def handler(request):
            """겹치는 모의 요청 수를 측정하고 짧게 응답을 지연."""
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return httpx.Response(200, json=completion_body())

        transport = AsyncRecordingTransport(recorder, httpx.MockTransport(handler))
        async with AsyncOpenAI(api_key="offline-dummy-secret", base_url="https://offline.invalid/v1", max_retries=0,
                               http_client=httpx.AsyncClient(transport=transport)) as client:
            async def invoke(question_id):
                """질문별 독립 논리 호출 범위를 열어 모의 SDK를 호출."""
                with recorder.scope("test_parallel_metric", question_id):
                    return await client.chat.completions.create(
                        model=EVALUATION_MODEL, messages=[{"role": "user", "content": question_id}],
                    )

            await asyncio.gather(*(invoke(f"Q{number:02d}") for number in range(1, 4)))

        self.assertEqual(peak, 1)
        self.assertEqual(len(recorder.calls), 3)
        self.assertEqual(len({call["logical_call_id"] for call in recorder.calls}), 3)
        self.assertEqual({call["question_id"] for call in recorder.calls}, {"Q01", "Q02", "Q03"})
        self.assertTrue(all(call["attempt"] == 1 for call in recorder.calls))


if __name__ == "__main__":
    unittest.main()
