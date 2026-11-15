"""실제 HTTP 대신 모의 응답으로 평가 연결·적용 범위·오류 기록을 검증한다."""

import asyncio
import json
import unittest

import httpx
from openai import AsyncOpenAI

from common.api import AsyncRecordingTransport, CallRecorder
from config import DATASET_PATH, EVALUATION_MODEL
from evaluation.judge import evaluate_judge
from evaluation.ragas_metrics import RecordedRagasLLM, evaluate_metrics, is_pure_abstention
from evaluation.rules import evaluate_rules
from schemas import GoldenDataset, JudgeEvaluation, RagasResults, RuleResult


def fixture(question, response="주당 평가가액은 32,783원입니다. [E1]"):
    """정식 결과 폴더를 만들지 않는 최소 RAG 모의 기록을 반환한다."""
    hits = [{"label": "E1", "chunk_id": "synthetic", "returned_order": 1,
             "rank_in_revision": 1, "distance": 0.1, "metadata": {"revision": "after"},
             "text": "주당 평가가액은 32,783원입니다."}]
    evidence = [{key: hit[key] for key in ("label", "metadata", "text")} for hit in hits]
    return {"question": question.question,
            "retrieval": {"execution_status": "completed", "results": hits},
            "generation": {"execution_status": "completed", "response": response,
                           "contexts": hits, "input": json.dumps({"question": question.question, "evidence": evidence}, ensure_ascii=False)}}


def mock_chat(content):
    """SDK가 정상적으로 역직렬화할 모의 Chat Completions 응답을 만든다."""
    return {"id": "chatcmpl-offline", "object": "chat.completion", "created": 0,
            "model": EVALUATION_MODEL, "service_tier": "default",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(content)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                      "prompt_tokens_details": {"cached_tokens": 0}}}


def offline_client(recorder, handler):
    """외부 네트워크에 연결하지 않는 SDK 클라이언트를 구성한다."""
    transport = AsyncRecordingTransport(recorder, httpx.MockTransport(handler))
    return AsyncOpenAI(api_key="offline-test-key", max_retries=0,
                       http_client=httpx.AsyncClient(transport=transport))


class EvaluatorTests(unittest.IsolatedAsyncioTestCase):
    """모의 API로 네 지표와 Judge의 실제 라이브러리 연결을 확인한다."""

    def setUp(self):
        """고정 골든 문항만 읽고 API 키·실제 응답은 읽지 않는다."""
        self.questions = GoldenDataset.model_validate_json(DATASET_PATH.read_text()).questions

    async def test_native_metrics_and_parameters(self):
        """기본 Ragas 계산과 평가 설정·출력 근거·논리 호출 분리를 검증한다."""
        recorder = CallRecorder()

        def handler(request):
            """지표가 요청한 출력 스키마에 맞는 합성 답변을 돌려준다."""
            body = json.loads(request.content)
            if request.url.path.endswith("/embeddings"):
                self.assertEqual(body["dimensions"], 3072)
                self.assertEqual(body["model"], "text-embedding-3-large")
                return httpx.Response(200, json={"object": "list", "model": body["model"],
                    "data": [{"object": "embedding", "index": i, "embedding": [1.0] + [0.0] * 3071} for i in range(len(body["input"]))],
                    "usage": {"prompt_tokens": 8, "total_tokens": 8}})
            self.assertEqual(body["model"], EVALUATION_MODEL)
            self.assertEqual(body["max_completion_tokens"], 8192)
            self.assertEqual(body["reasoning_effort"], "medium")
            self.assertEqual(body["service_tier"], "default")
            self.assertIs(body["store"], False)
            for key in ("temperature", "top_p", "max_tokens", "max_retries"):
                self.assertNotIn(key, body)
            messages = json.dumps(body["messages"])
            if "ContextPrecisionOutput" in messages:
                output = {"reason": "모의 근거가 유용합니다.", "verdict": 1}
            elif "ContextRecallOutput" in messages:
                output = {"classifications": [{"statement": "모의 사실", "reason": "근거 존재", "attributed": 1}]}
            elif "StatementGeneratorOutput" in messages:
                output = {"statements": ["주당 평가가액은 32,783원입니다."]}
            elif "NLIStatementOutput" in messages:
                output = {"statements": [{"statement": "주당 평가가액은 32,783원입니다.", "reason": "근거 존재", "verdict": 1}]}
            elif "AnswerRelevanceOutput" in messages:
                output = {"question": "주당 평가가액은 얼마인가요?", "noncommittal": 0}
            else:
                self.fail("알 수 없는 Ragas 출력 스키마")
            return httpx.Response(200, json=mock_chat(output))

        async with offline_client(recorder, handler) as client:
            results = await evaluate_metrics(self.questions[0], fixture(self.questions[0]), client, recorder)
        RagasResults.model_validate(results)
        self.assertEqual({x["evaluation_status"] for x in results.values()}, {"completed"})
        self.assertAlmostEqual(results["context_precision"]["score"], 1.0)
        self.assertEqual(results["answer_relevancy"]["score"], 1.0)
        self.assertEqual(len(results["answer_relevancy"]["details"]["llm_steps"]), 3)
        steps = results["answer_relevancy"]["details"]["llm_steps"] + results["answer_relevancy"]["details"]["embedding_steps"]
        linked_ids = [call_id for step in steps for call_id in step["call_ids"]]
        self.assertEqual(len(linked_ids), len(set(linked_ids)))
        self.assertEqual(set(linked_ids), set(results["answer_relevancy"]["call_ids"]))
        self.assertTrue(all(x["details"]["llm_steps"] for x in results.values()))
        self.assertEqual(len(recorder.calls), len({x["logical_call_id"] for x in recorder.calls}))

    async def test_concurrent_adapter_calls_keep_trace_ids_separate(self):
        """어댑터를 병렬 호출해도 내부 trace에 다른 호출 ID가 섞이지 않는다."""
        from ragas.metrics.collections.faithfulness.util import StatementGeneratorOutput
        recorder = CallRecorder()

        async def handler(request):
            """전송 중 태스크 전환을 허용해 동시에 시작한 호출을 모의한다."""
            await asyncio.sleep(0)
            return httpx.Response(200, json=mock_chat({"statements": ["모의 사실"]}))

        async with offline_client(recorder, handler) as client:
            llm = RecordedRagasLLM(client, recorder, "Q01")
            llm.metric_name = "test"
            await asyncio.gather(*(llm.agenerate("모의 프롬프트", StatementGeneratorOutput) for _ in range(3)))
        ids = [call_id for step in llm.trace for call_id in step["call_ids"]]
        self.assertEqual([len(step["call_ids"]) for step in llm.trace], [1, 1, 1])
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {call["call_id"] for call in recorder.calls})

    async def test_policy_and_pure_refusal_do_not_call_api(self):
        """답변 불가 문항의 고정 N/A와 순수 거절을 API 호출 없이 구분한다."""
        recorder = CallRecorder()
        question = self.questions[6]

        def handler(request):
            """금지된 모의 호출도 즉시 테스트 실패로 만든다."""
            self.fail("N/A 지표에서 호출했습니다.")

        async with offline_client(recorder, handler) as client:
            results = await evaluate_metrics(question, fixture(question, "제공된 근거에서는 확인할 수 없습니다."), client, recorder)
        RagasResults.model_validate(results)
        self.assertEqual({x["evaluation_status"] for x in results.values()}, {"not_applicable"})
        self.assertEqual(recorder.calls, [])
        self.assertFalse(is_pure_abstention("기관 경쟁률은 840.86대 1입니다. 제공된 근거에서는 실제 일반 경쟁률을 확인할 수 없습니다."))

    async def test_empty_claim_extraction_is_error(self):
        """사실을 쓴 답변에서 주장 추출이 비면 N/A가 아닌 오류로 남긴다."""
        recorder = CallRecorder()
        question = self.questions[6]

        def handler(request):
            """빈 주장 추출을 모의한다."""
            return httpx.Response(200, json=mock_chat({"statements": []}))

        async with offline_client(recorder, handler) as client:
            results = await evaluate_metrics(question, fixture(question), client, recorder)
        self.assertEqual(results["faithfulness"]["evaluation_status"], "error")
        self.assertIsNone(results["faithfulness"]["score"])
        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(results["faithfulness"]["details"]["llm_steps"][0]["output"], {"statements": []})

    async def test_invalid_json_is_not_retried_by_instructor(self):
        """출력 형식 오류는 한 번의 논리 평가로 끝내고 원본을 보존한다."""
        recorder = CallRecorder()

        def handler(request):
            """필수 statements가 없는 출력으로 Pydantic 오류를 유도한다."""
            return httpx.Response(200, json=mock_chat({"invalid": True}))

        question = self.questions[6]
        async with offline_client(recorder, handler) as client:
            results = await evaluate_metrics(question, fixture(question), client, recorder)
        self.assertEqual(results["faithfulness"]["evaluation_status"], "error")
        self.assertEqual(len(recorder.calls), 1)
        self.assertIsNotNone(recorder.calls[0]["response_body"])

    async def test_judge_quote_validation_preserves_raw(self):
        """스키마는 맞지만 존재하지 않는 인용을 쓴 Judge를 오류로 저장한다."""
        recorder = CallRecorder()
        assessment = {"verdict": "pass", "reason": "모의 이유", "response_quotes": ["없는 인용"], "basis_refs": ["C1"]}
        output = {"behavior": {"observed": "answer", "reason": "모의 이유"},
                  **{name: assessment for name in ("accuracy", "citation_appropriateness", "answer_scope")}}

        def handler(request):
            """Responses의 Structured Outputs 모양을 그대로 모의한다."""
            body = json.loads(request.content)
            self.assertTrue(body["text"]["format"]["strict"])
            self.assertEqual(set(json.loads(body["input"])), {"question", "expected_answerability", "golden_answer", "golden_criteria", "golden_sources", "generation_contexts", "response"})
            return httpx.Response(200, json={"id": "resp-offline", "object": "response", "created_at": 0,
                "model": EVALUATION_MODEL, "status": "completed",
                "output": [{"id": "msg-offline", "type": "message", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": json.dumps(output), "annotations": []}]}]})

        async with offline_client(recorder, handler) as client:
            result = await evaluate_judge(self.questions[0], fixture(self.questions[0]), client, recorder)
        JudgeEvaluation.model_validate(result)
        self.assertEqual(result["evaluation_status"], "error")
        self.assertIsNone(result["result"])
        self.assertEqual(json.loads(result["raw_response"]), output)

    def test_rules_bind_target_and_check_arithmetic(self):
        """숫자 포함만으로 통과시키지 않고 명시적 값·계산·인용 오류를 검사한다."""
        question = self.questions[0]
        records = evaluate_rules(question, fixture(question, "주당 평가가액은 31,000원입니다. [E99]\n할인율: (32,783 - 26,500) / 32,783 × 100 = 19.17%"))
        for record in records:
            RuleResult.model_validate(record)
        self.assertEqual(records[0]["verdict"], "fail")
        self.assertEqual(records[1]["verdict"], "undetermined")
        self.assertEqual(records[3]["verdict"], "pass")
        self.assertEqual(records[4]["verdict"], "fail")
        q9 = self.questions[8]
        records = evaluate_rules(q9, fixture(q9, "청약증거금: 100주 × 26,500원 × 50% = 1,325,000원 [E1]"))
        self.assertEqual(records[2]["verdict"], "pass")
        records = evaluate_rules(question, fixture(question, "주당 평가가액은 32,783원이 아닙니다."))
        self.assertEqual(records[0]["verdict"], "undetermined")


if __name__ == "__main__":
    unittest.main()
