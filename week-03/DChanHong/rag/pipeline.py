"""기존 검색·생성 동작으로 10문항을 실행하고 실제 입력과 답변을 보존한다."""

import hashlib
import json
import time
from pathlib import Path

from common.api import CallRecorder, make_client
from common.utils import new_id, now_iso, sha256_file, write_json
from common.validation import environment_info, load_dataset, validate_versions
from config import (
    ANSWER_MODEL,
    API_MAX_RETRIES,
    API_TIMEOUT,
    BASE_DIR,
    CHUNKS_PATH,
    COLLECTION_NAME,
    DATASET_PATH,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    INDEX_PATH,
    MAX_OUTPUT_TOKENS,
    PRICE_SNAPSHOT,
    TOP_K_PER_REVISION,
)
from .index import validate_index
from schemas import RagManifest, RagRecord
from prompts.generation import GENERATION_PROMPT, GENERATION_PROMPT_VERSION


def retrieve(collection, query_vector: list[float]) -> list[dict]:
    """정정 전 상위 3개 뒤에 정정 후 상위 3개를 연결한다."""
    evidence = []
    for revision in ("before", "after"):
        result = collection.query(
            query_embeddings=[query_vector],
            where={"revision": revision},
            n_results=TOP_K_PER_REVISION,
            include=["documents", "metadatas", "distances"],
        )
        for position, chunk_id in enumerate(result["ids"][0]):
            order = len(evidence) + 1
            evidence.append({
                "label": f"E{order}",
                "chunk_id": chunk_id,
                "returned_order": order,
                "rank_in_revision": position + 1,
                "distance": float(result["distances"][0][position]),
                "metadata": result["metadatas"][0][position],
                "text": result["documents"][0][position],
            })
    return evidence


def generation_input(question: str, evidence: list[dict]) -> str:
    """질문과 실제 검색 근거만 기존 입력 형식으로 직렬화한다."""
    return json.dumps({
        "question": question,
        "evidence": [
            {"label": item["label"], "metadata": item["metadata"], "text": item["text"]}
            for item in evidence
        ],
    }, ensure_ascii=False)


def pending_record(run_id: str, question_id: str, question: str) -> dict:
    """아직 호출하지 않은 문항의 모든 필드를 명시적으로 준비한다."""
    return {
        "schema_version": "rag_record_v1",
        "rag_run_id": run_id,
        "question_id": question_id,
        "question": question,
        "execution_status": "pending",
        "stage": "pending",
        "retrieval": {"execution_status": "pending", "results": [], "elapsed_ms": None, "error": None},
        "generation": {
            "execution_status": "pending", "instructions": None, "input": None,
            "contexts": [], "response": None, "api_response": None,
            "elapsed_ms": None, "error": None,
        },
        "api_calls": [],
        "started_at": None,
        "finished_at": None,
        "elapsed_ms": None,
    }


def save_record(path: Path, record: dict) -> None:
    """Pydantic 검증을 통과한 문항 기록만 저장한다."""
    validated = RagRecord.model_validate_json(json.dumps(record, ensure_ascii=False, allow_nan=False))
    write_json(path, validated.model_dump(mode="json"))


def _error_info(error: BaseException) -> dict:
    """민감한 요청 문자열 없이 오류 종류와 HTTP 상태를 남긴다."""
    code = type(error).__name__
    status = getattr(error, "status_code", None)
    return {"code": code, "message": f"{code}" + (f" (HTTP {status})" if status else "")}


def process_question(record: dict, collection, client, recorder: CallRecorder, instructions: str, path: Path) -> dict:
    """한 문항을 한 번 실행하며 오류나 빈 응답도 원본 그대로 보존한다."""
    started = time.perf_counter()
    excluded_write_seconds = 0.0
    first_call = len(recorder.calls)
    record.update(execution_status="running", stage="embedding", started_at=now_iso())
    record["retrieval"]["execution_status"] = "running"
    retrieval_started = time.perf_counter()
    generation_started = None
    try:
        with recorder.scope("rag_embedding", record["question_id"]):
            embedding = client.embeddings.create(
                model=EMBEDDING_MODEL,
                dimensions=EMBEDDING_DIMENSIONS,
                input=record["question"],
                encoding_format="float",
            )
        vector = embedding.data[0].embedding
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError("질문 임베딩 차원이 다릅니다.")
        record["stage"] = "retrieval"
        evidence = retrieve(collection, vector)
        request_input = generation_input(record["question"], evidence)
        record["retrieval"].update(
            execution_status="completed", results=evidence,
            elapsed_ms=(time.perf_counter() - retrieval_started) * 1000,
        )
        record["generation"].update(
            instructions=instructions, input=request_input, contexts=evidence,
        )
        record["stage"] = "generation"
        record["api_calls"] = list(recorder.calls[first_call:])
        write_started = time.perf_counter()
        save_record(path, record)
        excluded_write_seconds += time.perf_counter() - write_started
        record["generation"]["execution_status"] = "running"
        generation_started = time.perf_counter()
        with recorder.scope("rag_generation", record["question_id"]):
            response = client.responses.create(
                model=ANSWER_MODEL,
                instructions=instructions,
                input=request_input,
                reasoning={"effort": "none"},
                max_output_tokens=MAX_OUTPUT_TOKENS,
                store=False,
                service_tier="default",
            )
        record["generation"]["api_response"] = response.model_dump(mode="json")
        record["generation"]["response"] = response.output_text
        complete = response.status == "completed" and bool(response.output_text.strip())
        status = "completed" if complete else "needs_check"
        record["generation"].update(
            execution_status=status,
            elapsed_ms=(time.perf_counter() - generation_started) * 1000,
        )
        record.update(execution_status=status, stage="finished")
    except (Exception, KeyboardInterrupt) as error:
        info = _error_info(error)
        record["execution_status"] = "error"
        if record["retrieval"]["execution_status"] != "completed":
            record["retrieval"].update(
                execution_status="error", error=info,
                elapsed_ms=(time.perf_counter() - retrieval_started) * 1000,
            )
            record["generation"].update(execution_status="skipped", elapsed_ms=0)
        else:
            record["generation"].update(
                execution_status="error", error=info,
                elapsed_ms=(time.perf_counter() - generation_started) * 1000 if generation_started is not None else 0,
            )
        if isinstance(error, KeyboardInterrupt):
            raise
    finally:
        record.update(
            finished_at=now_iso(),
            elapsed_ms=(time.perf_counter() - started - excluded_write_seconds) * 1000,
            api_calls=list(recorder.calls[first_call:]),
        )
        save_record(path, record)
    return record


def build_manifest(run_id: str, dataset, instructions: str) -> dict:
    """문서·검색·모델·프롬프트와 재현에 필요한 환경을 고정 기록한다."""
    return {
        "schema_version": "rag_manifest_v1",
        "rag_run_id": run_id,
        "dataset": {"dataset_id": dataset.dataset_id, "path": str(DATASET_PATH), "sha256": sha256_file(DATASET_PATH)},
        "question_ids": [item.question_id for item in dataset.questions],
        "started_at": now_iso(),
        "finished_at": None,
        "code": environment_info(),
        "documents": [item.model_dump(mode="json") for item in dataset.documents],
        "chunking": {
            "path": str(CHUNKS_PATH), "sha256": sha256_file(CHUNKS_PATH),
            "max_chars": 1200, "overlap_chars": 200,
            "split_strategy": "줄 경계, 페이지·정정 구간별 분할; 2주차 원본 청크 재사용",
        },
        "retrieval": {
            "collection": COLLECTION_NAME, "distance_metric": "cosine",
            "top_k_per_revision": TOP_K_PER_REVISION,
            "revision_order": ["before", "after"], "index_path": str(INDEX_PATH),
        },
        "models": {
            "embedding": {"model": EMBEDDING_MODEL, "settings": {"dimensions": EMBEDDING_DIMENSIONS, "encoding_format": "float", "sdk_timeout_seconds": API_TIMEOUT, "sdk_max_retries": API_MAX_RETRIES}},
            "generation": {"model": ANSWER_MODEL, "settings": {"reasoning": {"effort": "none"}, "max_output_tokens": MAX_OUTPUT_TOKENS, "store": False, "service_tier": "default", "sdk_timeout_seconds": API_TIMEOUT, "sdk_max_retries": API_MAX_RETRIES}},
        },
        "prompt": {
            "version": GENERATION_PROMPT_VERSION, "text": instructions,
            "sha256": hashlib.sha256(instructions.encode("utf-8")).hexdigest(),
        },
        "pricing": PRICE_SNAPSHOT,
    }


def run_generation(output_dir: Path | None = None) -> Path:
    """Q01~Q10을 새 실행 폴더에 기록하며 사용자 요청으로만 API를 호출한다."""
    validate_versions()
    dataset = load_dataset()
    questions = [(item.question_id, item.question) for item in dataset.questions]
    collection = validate_index()
    instructions = GENERATION_PROMPT
    run_id = new_id("rag")
    run_dir = (Path(output_dir) if output_dir is not None else BASE_DIR / "output/rag_runs") / run_id
    manifest = build_manifest(run_id, dataset, instructions)
    del dataset  # 생성 루프에는 골든 답변·조건 객체를 전달하지 않는다.
    validated = RagManifest.model_validate_json(json.dumps(manifest, ensure_ascii=False, allow_nan=False))
    records = [pending_record(run_id, question_id, question) for question_id, question in questions]
    recorder = CallRecorder()
    with make_client(recorder) as client:
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "manifest.json", validated.model_dump(mode="json"))
        for record in records:
            save_record(run_dir / f"{record['question_id']}.json", record)
        print(f"실행 기록: {run_dir}", flush=True)
        try:
            for record in records:
                process_question(record, collection, client, recorder, instructions, run_dir / f"{record['question_id']}.json")
                print(f"{record['question_id']}: {record['execution_status']}", flush=True)
        finally:
            manifest["finished_at"] = now_iso()
            validated = RagManifest.model_validate_json(json.dumps(manifest, ensure_ascii=False, allow_nan=False))
            write_json(run_dir / "manifest.json", validated.model_dump(mode="json"))
    return run_dir
