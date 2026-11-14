"""저장된 답변을 고정 Judge 프롬프트와 출력 형식으로 평가한다."""

import json
import time

from schemas import GoldenQuestion, JudgeResult
from config import EVALUATION_MODEL, EVALUATION_MAX_OUTPUT_TOKENS
from prompts.judge import JUDGE_PROMPT


def judge_input(question: GoldenQuestion, rag: dict) -> dict:
    """Judge에 허용한 골든 자료·실제 컨텍스트·답변만 선택한다."""
    golden = question.model_dump(mode="json")
    return {
        key: golden[key]
        for key in (
            "question", "expected_answerability", "golden_answer",
            "golden_criteria", "golden_sources",
        )
    } | {
        "generation_contexts": rag["generation"]["contexts"],
        "response": rag["generation"]["response"],
    }


def validate_judge_references(result: JudgeResult, inputs: dict) -> None:
    """Judge가 실제 답변의 구절과 입력에 있는 C/G/E만 참조했는지 확인한다."""
    refs = {x["criterion_id"] for x in inputs["golden_criteria"]}
    refs.update(x["source_id"] for x in inputs["golden_sources"])
    refs.update(x["label"] for x in inputs["generation_contexts"])
    for name in ("accuracy", "citation_appropriateness", "answer_scope"):
        assessment = getattr(result, name)
        if not assessment.reason.strip():
            raise ValueError(f"{name}: 판정 이유가 비어 있습니다.")
        if any(not quote or quote not in inputs["response"]
               for quote in assessment.response_quotes):
            raise ValueError(f"{name}: 실제 답변에 없는 인용문입니다.")
        if not set(assessment.basis_refs) <= refs:
            raise ValueError(f"{name}: 입력에 없는 C/G/E 식별자입니다.")
    if not result.behavior.reason.strip():
        raise ValueError("behavior: 분류 이유가 비어 있습니다.")


async def evaluate_judge(question: GoldenQuestion, rag: dict, client, recorder) -> dict:
    """문항당 Judge를 한 번 호출하고 원문·검증 결과·호출 식별자를 보존한다."""
    record = {
        "evaluation_status": "skipped", "result": None, "raw_response": None,
        "call_ids": [], "elapsed_ms": 0, "error": None,
    }
    generation = rag["generation"]
    if generation["execution_status"] != "completed" or not generation["response"]:
        return record
    started = time.perf_counter()
    call_start = len(recorder.calls)
    try:
        inputs = judge_input(question, rag)
        with recorder.scope("judge", question.question_id):
            response = await client.responses.create(
                model=EVALUATION_MODEL,
                instructions=JUDGE_PROMPT,
                input=json.dumps(inputs, ensure_ascii=False),
                reasoning={"effort": "medium"}, max_output_tokens=EVALUATION_MAX_OUTPUT_TOKENS,
                store=False, service_tier="default",
                text={"format": {
                    "type": "json_schema", "name": "judge_result_v1", "strict": True,
                    "schema": JudgeResult.model_json_schema(),
                }},
            )
        record["raw_response"] = response.output_text
        if response.status != "completed":
            raise ValueError(f"Judge 응답 상태: {response.status}")
        result = JudgeResult.model_validate_json(response.output_text)
        validate_judge_references(result, inputs)
        record.update(evaluation_status="completed", result=result.model_dump(mode="json"))
    except Exception as exc:
        record.update(evaluation_status="error", error={
            "code": type(exc).__name__, "message": str(exc),
        })
    finally:
        record["call_ids"] = [call["call_id"] for call in recorder.calls[call_start:]]
        record["elapsed_ms"] = (time.perf_counter() - started) * 1000
    return record
