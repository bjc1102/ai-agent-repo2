"""유료 호출 없이 전체 10문항의 생성→평가→파일 기록 연결을 검증."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI, OpenAI

from common.api import AsyncRecordingTransport, RecordingTransport
from evaluation.runner import run_evaluation
from evaluation.ragas_metrics import blank_metric
from rag.pipeline import run_generation
from reporting.summary import load_records
from test_rag import FakeCollection


def generation_http(request):
    """실제 SDK가 해석할 수 있는 모의 임베딩·응답 JSON을 반환."""
    body = json.loads(request.content)
    if request.url.path.endswith("embeddings"):
        return httpx.Response(200, json={"object": "list", "model": body["model"],
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1] * 1536}],
            "usage": {"prompt_tokens": 10, "total_tokens": 10}})
    return httpx.Response(200, json={"id": "resp_offline", "object": "response", "created_at": 1,
        "model": body["model"], "status": "completed", "service_tier": "default",
        "output": [{"id": "msg_test", "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "모의 답변 [E1]", "annotations": []}]}],
        "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                  "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}})


def generation_client(recorder):
    """로컬 모의 HTTP 전송기에만 연결된 동기 SDK를 생성."""
    return OpenAI(api_key="offline-test", max_retries=0,
        http_client=httpx.Client(transport=RecordingTransport(recorder, httpx.MockTransport(generation_http))))


def evaluation_client(recorder):
    """평가 실행기에서 실수로 외부 요청을 할 수 없는 비동기 SDK를 생성."""
    def reject(request):
        """이 연결 테스트에서 예상하지 않은 평가 API 요청을 거부."""
        raise AssertionError("평가 함수 모의 연결을 벗어난 호출")
    return AsyncOpenAI(api_key="offline-test", max_retries=0,
        http_client=httpx.AsyncClient(transport=AsyncRecordingTransport(recorder, httpx.MockTransport(reject))))


async def metrics_result(question, rag, client, recorder):
    """정상 점수와 고정 N/A의 저장 연결만 시험하는 모의 지표 결과."""
    results = {}
    for name, policy in question.metric_policy.model_dump().items():
        if policy["mode"] == "not_applicable":
            results[name] = blank_metric("not_applicable", "metric_policy", policy["reason"])
        else:
            results[name] = {"evaluation_status": "completed", "score": 0.5,
                "reason_code": "offline_test", "reason": "기록 연결 검증용 모의 점수",
                "inputs": {"user_input": question.question}, "details": {"mock": True},
                "call_ids": [], "elapsed_ms": 1, "error": None}
    return results


async def judge_result(question, rag, client, recorder):
    """Judge의 보류 판정도 정상 실행 결과로 저장되는지 확인."""
    result = {"behavior": {"observed": "unclear", "reason": "연결 검증"},
        **{name: {"verdict": "needs_review", "reason": "모의 결과", "response_quotes": [], "basis_refs": []}
           for name in ("accuracy", "citation_appropriateness", "answer_scope")}}
    return {"evaluation_status": "completed", "result": result, "raw_response": json.dumps(result),
            "call_ids": [], "elapsed_ms": 1, "error": None}


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    """정식 출력 폴더와 분리된 임시 경로에서 전체 연결을 확인."""

    async def test_generate_then_evaluate_ten_saved_records(self):
        """10문항 저장·원본 복사·N/A·미검토·비용 분리·CSV 열 계약을 확인."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("rag.pipeline.make_client", generation_client), patch("rag.pipeline.validate_index", return_value=FakeCollection()):
                rag_dir = run_generation(root / "rag_runs")
            with patch("evaluation.runner.make_async_client", evaluation_client), patch("evaluation.runner.evaluate_metrics", metrics_result), patch("evaluation.runner.evaluate_judge", judge_result):
                eval_dir = await run_evaluation(rag_dir, root / "eval_runs")
            records = load_records(eval_dir)
            self.assertEqual(len(records), 10)
            self.assertTrue(all(r["rag"]["execution_status"] == "completed" for r in records))
            self.assertTrue(all(len(r["rag"]["api_calls"]) == 2 for r in records))
            self.assertTrue(all(r["human_reviews"] == [] and r["derived"]["human_final"] is None for r in records))
            self.assertEqual(records[6]["ragas"]["context_precision"]["evaluation_status"], "not_applicable")
            self.assertTrue(all(r["derived"]["rag_cost_usd"] > 0 and r["derived"]["eval_cost_usd"] == 0 for r in records))
            with (eval_dir / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 10)
            self.assertEqual(rows[6]["context_precision"], "")
            self.assertEqual(rows[0]["human_final"], "")
            summary = json.loads((eval_dir / "summary.json").read_text())
            self.assertEqual(summary["metrics"]["context_precision"]["valid_count"], 8)
            self.assertEqual(summary["metrics"]["answer_relevancy"]["valid_count"], 6)
