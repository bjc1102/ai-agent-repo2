"""저장된 RAG 결과만 평가하여 독립적인 새 실행에 기록."""

import hashlib
import json
import time
from pathlib import Path

from common.api import CallRecorder, make_async_client
from common.utils import json_hash, new_id, now_iso, sha256_file, write_json
from common.validation import environment_info, load_dataset, validate_versions
from config import BASE_DIR, DATASET_PATH, EVALUATION_MODEL, EVALUATION_EMBEDDING_MODEL, PRICE_SNAPSHOT
from prompts.judge import JUDGE_PROMPT, JUDGE_PROMPT_VERSION
from .judge import evaluate_judge
from .ragas_metrics import METRIC_NAMES, evaluate_metrics
from reporting.summary import derive, save_evaluation, summarize
from .rules import evaluate_rules
from schemas import EvaluationManifest, JudgeResult, RagManifest, RagRecord


def pending_evaluation(eval_id: str, golden, rag: dict) -> dict:
    """미실행 문항도 빠짐없이 남도록 평가 기록의 필수 구조를 준비."""
    record = {
        "schema_version": "evaluation_record_v1", "eval_run_id": eval_id,
        "rag_run_id": rag["rag_run_id"], "question_id": golden.question_id,
        "golden": golden.model_dump(mode="json"), "rag": rag,
        "ragas": {name: {"evaluation_status": "pending", "score": None, "reason_code": None,
                          "reason": None, "inputs": None, "details": None, "call_ids": [],
                          "elapsed_ms": None, "error": None} for name in METRIC_NAMES},
        "rules": [], "judge": {"evaluation_status": "pending", "result": None,
            "raw_response": None, "call_ids": [], "elapsed_ms": None, "error": None},
        "human_reviews": [], "api_calls": [],
    }
    record["derived"] = derive(record, None)
    return record


def build_manifest(eval_id: str, rag_id: str, rag_paths: list[Path], dataset) -> dict:
    """평가에 사용한 원본 해시·모델·프롬프트·출력 스키마를 기록."""
    prompt = JUDGE_PROMPT
    schema = JudgeResult.model_json_schema()
    common_settings = {"sdk_timeout_seconds": 120, "sdk_max_retries": 2, "api_concurrency": 1, "cache": False}
    return {
        "schema_version": "evaluation_manifest_v1", "eval_run_id": eval_id, "rag_run_id": rag_id,
        "started_at": now_iso(), "finished_at": None, "code": environment_info(),
        "dataset": {"dataset_id": dataset.dataset_id, "path": str(DATASET_PATH), "sha256": sha256_file(DATASET_PATH)},
        "rag_inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in rag_paths],
        "settings": {"ragas_version": "0.4.3", "rules_version": "rules_v1",
            "metric_settings": {"answer_relevancy": {"strictness": 3}, "policy_source": "golden.metric_policy", "native_prompts": "문항 details.llm_steps에 프롬프트·스키마·해시 저장"}},
        "models": {
            "ragas_llm": {"model": EVALUATION_MODEL, "settings": {**common_settings, "reasoning_effort": "medium", "max_completion_tokens": 8192, "instructor_max_retries": 0, "store": False, "service_tier": "default"}},
            "embedding": {"model": EVALUATION_EMBEDDING_MODEL, "settings": {**common_settings, "dimensions": 3072}},
            "judge": {"model": EVALUATION_MODEL, "settings": {**common_settings, "reasoning": {"effort": "medium"}, "max_output_tokens": 8192, "store": False, "service_tier": "default"}},
        },
        "judge_prompt": {"version": JUDGE_PROMPT_VERSION, "text": prompt, "sha256": hashlib.sha256(prompt.encode()).hexdigest()},
        "judge_schema": {"version": "judge_result_v1", "json_schema": schema, "sha256": json_hash(schema)},
        "pricing": PRICE_SNAPSHOT,
    }


async def run_evaluation(rag_run_dir: Path, output_dir: Path | None = None) -> Path:
    """검색·생성을 재실행하지 않고 저장된 10문항을 한 번씩 평가."""
    validate_versions()
    dataset = load_dataset()
    rag_run_dir = Path(rag_run_dir).resolve()
    source_manifest = RagManifest.model_validate_json((rag_run_dir / "manifest.json").read_text(encoding="utf-8"))
    rag_paths = [rag_run_dir / f"{q.question_id}.json" for q in dataset.questions]
    rag_records = [RagRecord.model_validate_json(p.read_text(encoding="utf-8")).model_dump(mode="json") for p in rag_paths]
    for question, rag in zip(dataset.questions, rag_records):
        if (rag["question_id"] != question.question_id or rag["question"] != question.question
                or rag["rag_run_id"] != source_manifest.rag_run_id):
            raise ValueError("원래 RAG 질문·실행과 평가 입력이 다릅니다.")
    recorder = CallRecorder()
    client = make_async_client(recorder)
    run_id = new_id("eval")
    run_dir = (Path(output_dir) if output_dir else BASE_DIR / "output/eval_runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = build_manifest(run_id, source_manifest.rag_run_id, [rag_run_dir / "manifest.json", *rag_paths], dataset)
    validated = EvaluationManifest.model_validate_json(json.dumps(manifest, ensure_ascii=False))
    write_json(run_dir / "manifest.json", validated.model_dump(mode="json"))
    records = [pending_evaluation(run_id, q, r) for q, r in zip(dataset.questions, rag_records)]
    for record in records:
        save_evaluation(run_dir / f"{record['question_id']}.json", record)
    print(f"평가 기록: {run_dir}", flush=True)
    try:
        for question, record in zip(dataset.questions, records):
            started, call_start = time.perf_counter(), len(recorder.calls)
            try:
                record["rules"] = evaluate_rules(question, record["rag"])
                record["ragas"] = await evaluate_metrics(question, record["rag"], client, recorder)
                record["judge"] = await evaluate_judge(question, record["rag"], client, recorder)
            finally:
                record["api_calls"] = list(recorder.calls[call_start:])
                record["derived"] = derive(record, (time.perf_counter() - started) * 1000)
                save_evaluation(run_dir / f"{question.question_id}.json", record)
            print(f"{question.question_id}: Judge {record['judge']['evaluation_status']}", flush=True)
    finally:
        manifest["finished_at"] = now_iso()
        write_json(run_dir / "manifest.json", EvaluationManifest.model_validate_json(json.dumps(manifest)).model_dump(mode="json"))
        await client.close()
        summarize(run_dir)
    return run_dir
