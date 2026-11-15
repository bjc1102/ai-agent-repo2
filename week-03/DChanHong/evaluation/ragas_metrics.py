"""Ragas 기본 지표를 실행하고 내부 판정·실제 요청과 연결한다."""

import hashlib
import json
import math
import os
import re
import time

# 평가 자료가 프레임워크의 사용 통계로 전달되지 않도록 한다.
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

import instructor
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorLLM
from ragas.metrics.collections import (
    AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness,
)

from schemas import GoldenQuestion
from config import (
    EVALUATION_MODEL, EVALUATION_EMBEDDING_MODEL,
    EVALUATION_EMBEDDING_DIMENSIONS, EVALUATION_MAX_OUTPUT_TOKENS,
)


METRIC_NAMES = (
    "context_precision", "context_recall", "faithfulness", "answer_relevancy",
)


def json_hash(value) -> str:
    """프롬프트와 출력 스키마를 재현할 수 있도록 해시를 계산한다."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RecordedRagasLLM(InstructorLLM):
    """기본 Ragas 프롬프트를 유지하고 GPT-5.4 호출 설정·판정 근거를 기록한다."""

    def __init__(self, client, recorder, question_id: str):
        """JSON 모드의 Instructor와 캐시 없는 평가 클라이언트를 준비한다."""
        super().__init__(
            client=instructor.from_openai(client, mode=instructor.Mode.JSON),
            model=EVALUATION_MODEL, provider="openai", cache=None,
        )
        self.recorder = recorder
        self.question_id = question_id
        self.metric_name = "unassigned"
        self.trace = []

    def _map_openai_params(self) -> dict:
        """Ragas의 모델명 판별과 무관하게 확정한 API 파라미터만 전달한다."""
        return {
            "reasoning_effort": "medium", "max_completion_tokens": EVALUATION_MAX_OUTPUT_TOKENS,
            "store": False, "service_tier": "default", "max_retries": 0,
        }

    async def agenerate(self, prompt, response_model):
        """각 내부 호출의 프롬프트·스키마·검증된 출력·물리 호출 ID를 보존한다."""
        schema = response_model.model_json_schema()
        entry = {
            "prompt": prompt, "prompt_sha256": json_hash(prompt),
            "response_schema": schema, "response_schema_sha256": json_hash(schema),
            "output": None, "error": None, "call_ids": [],
        }
        self.trace.append(entry)
        scope = None
        try:
            with self.recorder.scope(f"ragas.{self.metric_name}.llm", self.question_id) as scope:
                result = await super().agenerate(prompt, response_model)
            entry["output"] = result.model_dump(mode="json")
            validate_internal_output(entry["output"])
            return result
        except Exception as exc:
            entry["error"] = {"code": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            entry["call_ids"] = [c["call_id"] for c in self.recorder.calls
                                 if scope is not None and c["logical_call_id"] == scope["logical_call_id"]]


class RecordedEmbeddings(BaseRagasEmbedding):
    """선택한 평가 임베딩의 차원을 고정하고 순차 호출과 계산 입력을 기록한다."""

    def __init__(self, client, recorder, question_id: str):
        """비동기 SDK와 문항 식별자를 보관하고 로컬 캐시는 사용하지 않는다."""
        super().__init__(cache=None)
        self.client = client
        self.recorder = recorder
        self.question_id = question_id
        self.trace = []

    def embed_text(self, text, **kwargs):
        """동기 호출의 우발적인 병렬 실행을 막고 비동기 경로를 안내한다."""
        raise RuntimeError("평가 임베딩은 aembed_text 또는 aembed_texts를 사용합니다.")

    async def aembed_text(self, text, **kwargs):
        """질문 하나의 3,072차원 벡터를 반환한다."""
        return (await self.aembed_texts([text]))[0]

    async def aembed_texts(self, texts, **kwargs):
        """하나의 배치 호출로 벡터와 순서를 검증하고 계산 근거를 남긴다."""
        with self.recorder.scope("ragas.answer_relevancy.embedding", self.question_id) as scope:
            response = await self.client.embeddings.create(
                model=EVALUATION_EMBEDDING_MODEL, dimensions=EVALUATION_EMBEDDING_DIMENSIONS,
                input=texts, encoding_format="float",
            )
        data = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in data] != list(range(len(texts))):
            raise ValueError("임베딩 응답 개수 또는 순서가 요청과 다릅니다.")
        vectors = [item.embedding for item in data]
        if any(len(vector) != EVALUATION_EMBEDDING_DIMENSIONS or not all(math.isfinite(x) for x in vector)
               for vector in vectors):
            raise ValueError("유효한 3,072차원 평가 임베딩이 아닙니다.")
        self.trace.append({
            "texts": list(texts), "vectors": vectors,
            "call_ids": [c["call_id"] for c in self.recorder.calls
                         if c["logical_call_id"] == scope["logical_call_id"]],
        })
        return vectors


def validate_internal_output(value) -> None:
    """Ragas 출력 스키마의 정수 판정이 실제 이진값인지 추가 확인한다."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("verdict", "attributed", "noncommittal") and item not in (0, 1):
                raise ValueError(f"Ragas 내부 {key}는 0 또는 1이어야 합니다.")
            validate_internal_output(item)
    elif isinstance(value, list):
        for item in value:
            validate_internal_output(item)


def is_pure_abstention(response: str) -> bool:
    """추가 사실 없이 확인 불가만 안내한 명확한 문장에 한해 참을 반환한다."""
    text = re.sub(r"\[E\d+\]", "", response)
    text = re.sub(r"[*_#]", "", text).strip()
    if not text or re.search(r"\d|지만|이므로|때문|이며|이고|반면|따라서|다만", text):
        return False
    sentences = [s.strip(" -\t") for s in re.split(r"[.!?\n]+", text) if s.strip(" -\t")]
    if not sentences:
        return False
    endings = (
        r"(?:(?:제공된|해당|주어진)\s*)?(?:문서|근거|자료)(?:만으로|에서는|에서|로는|에는|상)?"
        r"\s*(?:확인할 수 없(?:습니다|음)|확인이 불가(?:합니다|능합니다)|알 수 없습니다)"
    )
    # 임의의 앞 문장을 허용하지 않고 질문의 대상에 대한 짧은 주제어만 허용한다.
    pattern = re.compile(r"(?:[^,;:。]{1,65}?(?:은|는|을|를)\s+)?" + endings + r"[\s]*")
    return all(pattern.fullmatch(sentence) is not None for sentence in sentences)


def blank_metric(status: str, code: str, reason: str) -> dict:
    """실행하지 않는 지표를 점수 없이 사유와 함께 기록한다."""
    return {
        "evaluation_status": status, "score": None, "reason_code": code,
        "reason": reason, "inputs": None, "details": None,
        "call_ids": [], "elapsed_ms": 0, "error": None,
    }


def actual_contexts(rag: dict) -> list[str]:
    """저장된 실제 생성 요청에서 라벨·메타데이터·본문을 그대로 읽는다."""
    payload = json.loads(rag["generation"]["input"])
    if payload["question"] != rag["question"] or not isinstance(payload["evidence"], list):
        raise ValueError("실제 생성 요청과 문항의 연결을 확인할 수 없습니다.")
    return [json.dumps(item, ensure_ascii=False) for item in payload["evidence"]]


def retrieval_contexts(rag: dict) -> list[str]:
    """검색 당시 순서를 바꾸지 않고 컨텍스트와 제공 메타데이터를 묶는다."""
    return [json.dumps({key: hit[key] for key in ("label", "metadata", "text")},
                       ensure_ascii=False) for hit in rag["retrieval"]["results"]]


async def evaluate_metrics(question: GoldenQuestion, rag: dict, client, recorder) -> dict:
    """문항 정책에 맞는 네 지표를 순차 평가하며 N/A·실패와 점수를 구분한다."""
    llm = RecordedRagasLLM(client, recorder, question.question_id)
    embeddings = RecordedEmbeddings(client, recorder, question.question_id)
    metrics = {
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(
            llm=llm, embeddings=embeddings, strictness=3, allowed_values=(-1.0, 1.0),
        ),
    }
    results = {}
    for name in METRIC_NAMES:
        policy = getattr(question.metric_policy, name)
        if policy.mode == "not_applicable":
            results[name] = blank_metric("not_applicable", "metric_policy", policy.reason)
            continue
        retrieval_metric = name.startswith("context_")
        stage = rag["retrieval"] if retrieval_metric else rag["generation"]
        if stage["execution_status"] != "completed" or (
            not retrieval_metric and not stage["response"]
        ):
            results[name] = blank_metric("skipped", "upstream_not_completed", "필요한 RAG 단계가 정상 완료되지 않았습니다.")
            continue
        if name == "faithfulness" and is_pure_abstention(stage["response"]):
            results[name] = blank_metric("not_applicable", "pure_abstention", "추가 사실 주장 없이 확인 불가만 안내한 답변입니다.")
            continue
        started = time.perf_counter()
        call_start = len(recorder.calls)
        llm.trace = []
        embeddings.trace = []
        llm.metric_name = name
        record = blank_metric("pending", "pending", "평가 준비 중입니다.")
        try:
            inputs = {"user_input": question.question}
            if retrieval_metric:
                inputs.update(retrieved_contexts=retrieval_contexts(rag), reference=question.retrieval_reference)
            else:
                inputs["response"] = stage["response"]
                if name == "faithfulness":
                    inputs["retrieved_contexts"] = actual_contexts(rag)
            record["inputs"] = inputs
            result = await metrics[name].ascore(**inputs)
            score = float(result.value)
            lower = -1 if name == "answer_relevancy" else 0
            if not math.isfinite(score) or not lower <= score <= 1:
                raise ValueError("지표가 유효한 점수를 반환하지 않았습니다. 주장 추출·판정 원문을 확인하세요.")
            record.update(
                evaluation_status="completed", score=score, reason_code="ragas_internal_trace",
                reason="Ragas 기본 계산식으로 산출했습니다. details에 내부 판정·계산 입력을 보존했습니다.",
            )
        except Exception as exc:
            record.update(
                evaluation_status="error", score=None, reason_code="metric_error", reason=str(exc),
                error={"code": type(exc).__name__, "message": str(exc)},
            )
        finally:
            record["details"] = {
                "framework": "ragas", "version": "0.4.3",
                "metric_class": type(metrics[name]).__name__,
                "llm_steps": llm.trace, "embedding_steps": embeddings.trace,
            }
            record["call_ids"] = [call["call_id"] for call in recorder.calls[call_start:]]
            record["elapsed_ms"] = (time.perf_counter() - started) * 1000
        results[name] = record
    return results
