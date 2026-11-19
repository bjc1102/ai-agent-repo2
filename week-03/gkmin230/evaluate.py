import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile


HERE = Path(__file__).resolve().parent
PACKAGES = ["ragas", "chromadb", "openai", "python-dotenv", "langchain-openai"]
SYSTEM = (
    "너는 AWS GuardDuty 보안 분석 도우미다. "
    "반드시 제공된 참고 문서에 근거하여 한국어로 답변하라. "
    "참고 문서에 없는 사실이나 수치는 추측하지 마라. "
    "질문에 답할 근거가 부족하면 "
    "'제공된 문서만으로는 확인할 수 없습니다.'라고 답하라. "
    "검색 문서에 관련 단어가 있다는 이유만으로 답을 만들어내지 마라. "
    "여러 부분으로 된 질문은 근거가 있는 부분을 설명하고, "
    "확인할 수 없는 부분은 구분해서 밝혀라."
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def versions():
    found = {}
    for name in PACKAGES:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def validate(root, questions):
    source = (root / "data/guardduty.txt").read_text(encoding="utf-8")
    chunks = read_json(root / "data/chunks.json")
    assert len(questions) == 5
    assert len({item["id"] for item in questions}) == 5
    for item in questions:
        assert item["answerability"] in {"answerable", "unanswerable", "partial"}
        for evidence in item["evidence"] + item.get("scope_evidence", []):
            assert 1 <= evidence["start_line"] <= evidence["end_line"] <= len(source.splitlines())
        assert item["expected_answer"]
    matches = all(chunk["text"] == source[chunk["chunk_index"] * 700:chunk["chunk_index"] * 700 + 800] for chunk in chunks)
    return {"question_count": len(questions), "chunk_count": len(chunks),
            "fixed_800_overlap_100_matches_all_chunks": matches,
            "source_sha256": digest(root / "data/guardduty.txt"),
            "chunks_sha256": digest(root / "data/chunks.json"), "packages": versions()}


def messages(question, documents):
    context = "\n\n".join(f"[문서 {rank}]\n{doc['text']}" for rank, doc in enumerate(documents, 1))
    return context, [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"참고 문서:\n{context}\n\n질문: {question}"}]


def metric_record(status, reason):
    return {"status": status, "score": None, "reason": reason, "judge_outputs": []}


def safe_error(error):
    """Keep diagnostic types/errno, never exception text, URLs or credentials."""
    chain = []
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append({"type": type(error).__name__, "errno": getattr(error, "errno", None)})
        error = error.__cause__ or error.__context__
    return chain


def initial_metrics(item):
    metrics = {
        "context_precision": metric_record("not_run", "Ragas 평가 미실행"),
        "faithfulness": metric_record("not_run", "Ragas 평가 미실행")}
    if item["answerability"] == "unanswerable":
        metrics["context_precision"] = metric_record("N/A", "문서에 정답 근거가 없어 reference 기반 관련성 기준을 정의하지 않음. 거절 문장을 검색 정답으로 사용하지 않음.")
        metrics["faithfulness"] = metric_record("N/A", "이 문항은 외부 사실 대신 답변 거절 행동을 평가. 실질 사실 주장이 없는 거절의 Faithfulness를 성공 점수로 처리하지 않음. 수치 환각 여부는 별도 판정.")
    return metrics


def human_template(questions, rows, output):
    by_id = {row["id"]: row for row in rows}
    path = output / "human_review.json"
    if path.exists():
        return
    write_json(path, {"status": "pending_human_review", "instructions":
        "실제 사람이 원문과 답변을 비교한 뒤 reviewer, reviewed_at, verdict, reason을 작성. 자동 점수나 에이전트 분석을 사람 판정으로 복사하지 말 것. run_id가 다르면 새 판정 파일을 만들 것.",
        "result_sha256": digest(output / "evaluation_results.json"),
        "reviews": [{"id": item["id"], "question": item["question"],
                     "answer": by_id[item["id"]].get("answer"), "evidence": item["evidence"],
                     "expected_answer": item["expected_answer"], "reviewer": None, "reviewed_at": None,
                     "verdict": "pending", "source_support": None, "retrieval_found_evidence": None,
                     "generation_used_evidence": None, "answerability_handled": None,
                     "claim_comparisons": [], "reason": None} for item in questions]})


def offline(root, questions, output):
    checks = validate(root, questions)
    old = read_json(root / "results.json")
    previous = {row["question"]: row for row in old["experiments"]}
    chunks = {chunk["chunk_index"]: chunk for chunk in read_json(root / "data/chunks.json")}
    rows = []
    for item in questions:
        historical = previous.get(item["question"])
        row = {"id": item["id"], "question": item["question"], "status": "not_run",
               "retrieved_documents": [], "actual_context": None, "actual_messages": None,
               "answer": None, "metrics": initial_metrics(item)}
        if historical:
            documents = []
            for document in historical["retrieved_documents"]:
                document = dict(document)
                document["chunk_id"] = chunks[document["chunk_index"]]["id"]
                document["chunk_id_origin"] = "mapped_from_local_chunks_by_index"
                documents.append(document)
            row.update(status="historical_import_not_new_run", retrieved_documents=documents,
                       answer=historical["answer"], historical_created_at=old["created_at"],
                       reconstructed_context=messages(item["question"], documents)[0],
                       context_provenance="기존 results.json에 실제 컨텍스트 미기록. actual_context는 null이며 별도 값은 기존 rag.py 연결 규칙으로 재구성한 것.")
        rows.append(row)
    write_json(output / "evaluation_results.json", {"mode": "offline_historical_import", "checks": checks,
               "historical_file_sha256": digest(root / "results.json"), "experiments": rows})
    human_template(questions, rows, output)
    write_json(output / "validation.json", checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2))


def judge_handler():
    from langchain_core.callbacks import BaseCallbackHandler

    class JudgeCapture(BaseCallbackHandler):
        def __init__(self):
            self.outputs = []
            self.usage = []

        def on_llm_end(self, response, **kwargs):
            self.usage.append((response.llm_output or {}).get("token_usage", {}))
            for group in response.generations:
                for generation in group:
                    raw = generation.text
                    try:
                        parsed = json.loads(raw)
                    except (ValueError, TypeError):
                        parsed = None
                    self.outputs.append({"raw_output": raw, "parsed_output": parsed})

    return JudgeCapture()


async def score(metric, sample):
    required = metric.required_columns.get("SINGLE_TURN", set())
    values = sample.to_dict()
    missing = [name for name in required if not values.get(name)]
    if missing:
        result = metric_record("N/A", "필수 입력 누락: " + ", ".join(missing))
        result["required_inputs"] = sorted(required)
        return result
    capture = judge_handler()
    try:
        value = float(await metric.single_turn_ascore(sample, callbacks=[capture]))
        if math.isfinite(value):
            result = metric_record("scored", "공식 Ragas 점수. judge_outputs의 reason/verdict/statement를 함께 확인.")
            result["score"] = value
        else:
            result = metric_record("N/A", "Ragas 비유한 점수: 추출 가능한 사실 주장 부재 또는 파싱 실패. Judge 원문 확인 필요.")
    except Exception as error:
        result = metric_record("error", f"Ragas 평가 오류 유형: {type(error).__name__}. 비밀정보 보호를 위해 예외 본문 생략.")
        result["error_chain"] = safe_error(error)
    result.update(required_inputs=sorted(required), judge_outputs=capture.outputs,
                  judge_usage=capture.usage,
                  reason_capture="모델이 반환한 평가 설명만 기록하며 비공개 내부 추론은 수집하지 않음")
    return result


async def live(root, questions, output):
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    installed = versions()
    if installed["ragas"] != "0.3.5" or any(value is None for value in installed.values()):
        raise SystemExit("필수 패키지 미설치 또는 Ragas 버전 불일치. requirements.txt를 승인 후 설치하세요.")
    import chromadb
    from chromadb.config import Settings
    from dotenv import load_dotenv
    from openai import OpenAI
    from langchain_openai import ChatOpenAI
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference
    from ragas.run_config import RunConfig

    checks = validate(root, questions)
    try:
        socket.getaddrinfo("api.openai.com", 443)
    except OSError as error:
        write_json(output / "connectivity_check.json", {
            "status": "failed", "stage": "dns_resolution", "host": "api.openai.com",
            "error_chain": safe_error(error), "api_calls": 0,
            "reason": "실행 환경 DNS/네트워크 연결 필요. 패키지나 평가 API 호환성 오류가 아님."})
        raise SystemExit("api.openai.com DNS 조회 실패. connectivity_check.json 참조; API 실행 중단.")
    load_dotenv(root / ".env", override=False)
    client = OpenAI(max_retries=0, timeout=60)
    judge = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=2048, max_retries=0, timeout=60),
        run_config=RunConfig(timeout=60, max_retries=1))
    metrics = {"context_precision": LLMContextPrecisionWithReference(llm=judge),
               "faithfulness": Faithfulness(llm=judge)}
    report = {"mode": "live", "created_at": datetime.now(timezone.utc).isoformat(),
              "checks": checks, "config": {"embedding_model": "text-embedding-3-small",
              "answer_model": "gpt-4o-mini", "judge_model": "gpt-4o-mini", "temperature": 0,
              "judge_temperature": 0.01, "max_output_tokens": 2048, "api_max_retries": 0,
              "ragas_max_attempts": 1,
              "top_k": 3, "collection": "guardduty_docs", "db_mode": "temporary_copy"}, "experiments": []}
    with tempfile.TemporaryDirectory(prefix="week03_chroma_") as temporary:
        copied = Path(temporary) / "chroma_db"
        shutil.copytree(root / "chroma_db", copied)
        database = chromadb.PersistentClient(path=str(copied), settings=Settings(anonymized_telemetry=False))
        collection = database.get_collection("guardduty_docs")
        for item in questions:
            row = {"id": item["id"], "question": item["question"], "status": "started",
                   "retrieved_documents": [], "actual_context": None, "actual_messages": None, "answer": None,
                   "metrics": initial_metrics(item)}
            report["experiments"].append(row)
            stage = "embedding"
            try:
                embedded = client.embeddings.create(model="text-embedding-3-small", input=item["question"])
                stage = "retrieval"
                retrieved = collection.query(query_embeddings=[embedded.data[0].embedding], n_results=3,
                                             include=["documents", "distances", "metadatas"])
                documents = [{"rank": rank + 1, "chunk_id": identifier,
                              "distance": float(retrieved["distances"][0][rank]),
                              "metadata": retrieved["metadatas"][0][rank],
                              "text": retrieved["documents"][0][rank]}
                             for rank, identifier in enumerate(retrieved["ids"][0])]
                context, prompt = messages(item["question"], documents)
                row.update(retrieved_documents=documents, actual_context=context, actual_messages=prompt)
                stage = "generation"
                response = client.chat.completions.create(model="gpt-4o-mini", temperature=0, max_tokens=2048, messages=prompt)
                row.update(answer=response.choices[0].message.content or "", status="generated",
                           generation_usage=response.usage.model_dump() if response.usage else None,
                           embedding_usage=embedded.usage.model_dump())
                write_json(output / "evaluation_results.json", report)
                sample = SingleTurnSample(user_input=item["question"], response=row["answer"],
                                          retrieved_contexts=[document["text"] for document in documents],
                                          reference=item["reference"])
                stage = "evaluation"
                for name, metric in metrics.items():
                    if row["metrics"][name]["status"] != "N/A":
                        row["metrics"][name] = await score(metric, sample)
                row["evaluation_scope"] = item.get("reference_scope", "전체 문항; 거절 행동은 별도 사람 평가")
                row["status"] = "evaluation_error" if any(
                    entry["status"] == "error" for entry in row["metrics"].values()) else "completed"
            except Exception as error:
                row.update(status="error", error_type=type(error).__name__, error_stage=stage,
                           error_chain=safe_error(error))
                for name, entry in row["metrics"].items():
                    if entry["status"] == "not_run":
                        row["metrics"][name] = metric_record("not_run", f"{stage} 단계 오류로 평가 입력 미생성")
            write_json(output / "evaluation_results.json", report)
            print(item["id"], row["status"])
    human_template(questions, report["experiments"], output)
    if any(row["status"] != "completed" for row in report["experiments"]):
        raise SystemExit("일부 문항 실행/평가 실패. evaluation_results.json 참조.")


def main():
    parser = argparse.ArgumentParser(description="GuardDuty RAG 평가: 기본은 API 없는 오프라인 검사")
    parser.add_argument("--project-root", type=Path, default=HERE.parent)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--run-api", action="store_true", help="승인받은 경우에만 사용; OpenAI에 질문/컨텍스트/정답 전송")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output.resolve()
    if args.run_api and (output / "evaluation_results.json").exists():
        raise SystemExit("기존 결과 보존을 위해 새로운 --output 경로를 지정하세요.")
    questions = read_json(HERE / "questions.json")["questions"]
    if args.run_api:
        print("OpenAI API 호출: 임베딩 5회 + 생성 5회 + Judge 약 20회(재시도/파싱에 따라 증가 가능).")
        asyncio.run(live(root, questions, output))
    else:
        offline(root, questions, output)


if __name__ == "__main__":
    main()
