"""상세 평가 기록에서 파생값과 JSON·CSV 요약을 생성."""

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from common.api import aggregate_cost
from common.utils import write_json
from schemas import EvaluationRecord

METRICS = ("context_precision", "context_recall", "faithfulness", "answer_relevancy")
DIMENSIONS = ("accuracy", "citation_appropriateness", "answer_scope")
CSV_COLUMNS = [
    "eval_run_id", "rag_run_id", "question_id", "topic", "expected_answerability",
    "rag_execution_status", "rag_error_summary", "context_precision", "context_precision_status",
    "context_recall", "context_recall_status", "faithfulness", "faithfulness_status",
    "answer_relevancy", "answer_relevancy_status", "metric_notes", "rules_summary",
    "judge_status", "judge_accuracy", "judge_citation", "judge_scope", "judge_overall",
    "observed_behavior", "behavior_matches_expected", "human_review_status", "human_accuracy",
    "human_citation", "human_scope", "human_final", "disagreement_summary", "rag_cost_usd",
    "rag_cost_complete", "rag_elapsed_ms", "eval_cost_usd", "eval_cost_complete", "eval_elapsed_ms",
    "detail_json_path",
]


def derive(record: dict, elapsed_ms: float | None) -> dict:
    """독립적인 판정과 RAG·이번 평가 비용을 각각 계산."""
    result = record["judge"]["result"]
    overall = matches = None
    if result is not None:
        verdicts = [result[name]["verdict"] for name in DIMENSIONS]
        overall = "fail" if "fail" in verdicts else "needs_review" if "needs_review" in verdicts else "pass"
        observed = result["behavior"]["observed"]
        expected = {"answerable": "answer", "unanswerable": "abstain", "partial": "partial_answer"}
        matches = None if observed == "unclear" else observed == expected[record["golden"]["expected_answerability"]]
    latest = record["human_reviews"][-1] if record["human_reviews"] else None
    rag_cost, eval_cost = aggregate_cost(record["rag"]["api_calls"]), aggregate_cost(record["api_calls"])
    return {
        "behavior_matches_expected": matches, "judge_overall": overall,
        "rule_counts": dict(Counter(rule["verdict"] or rule["evaluation_status"] for rule in record["rules"])),
        "latest_human_review_id": latest["review_id"] if latest else None,
        "human_final": ("pass" if all(latest[d]["verdict"] == "pass" for d in DIMENSIONS) else "fail") if latest else None,
        "disagreement_summary": "; ".join(f"{d['component']}.{d['item']}: {d['reason']}" for d in latest["disagreements"]) if latest else "",
        "rag_cost_usd": rag_cost["cost_usd"], "rag_known_cost_usd": rag_cost["known_cost_usd"],
        "rag_cost_complete": rag_cost["cost_complete"], "rag_elapsed_ms": record["rag"]["elapsed_ms"],
        "eval_cost_usd": eval_cost["cost_usd"], "eval_known_cost_usd": eval_cost["known_cost_usd"],
        "eval_cost_complete": eval_cost["cost_complete"], "eval_elapsed_ms": elapsed_ms,
        "performed_metric_count": sum(m["evaluation_status"] in ("completed", "error") for m in record["ragas"].values()),
    }


def save_evaluation(path: Path, record: dict) -> None:
    """완전한 필드와 상태 계약을 검사한 평가 JSON을 저장."""
    validated = EvaluationRecord.model_validate_json(json.dumps(record, ensure_ascii=False, allow_nan=False))
    write_json(path, validated.model_dump(mode="json"))


def load_records(run_dir: Path) -> list[dict]:
    """10개 상세 파일을 빠짐없이 읽고 서로 같은 실행인지 확인."""
    records = []
    for number in range(1, 11):
        path = Path(run_dir) / f"Q{number:02d}.json"
        record = EvaluationRecord.model_validate_json(path.read_text(encoding="utf-8"))
        if record.question_id != f"Q{number:02d}":
            raise ValueError(f"파일명과 문항 ID가 다릅니다: {path.name}")
        records.append(record.model_dump(mode="json"))
    if len({r["eval_run_id"] for r in records}) != 1 or len({r["rag_run_id"] for r in records}) != 1:
        raise ValueError("요약 대상에 서로 다른 실행의 기록이 섞여 있습니다.")
    return records


def time_stats(values: list[float]) -> dict:
    """실제 완료 시간이 있는 표본만 평균·중앙값·최댓값을 계산."""
    return {"count": len(values), "mean_ms": statistics.mean(values) if values else None,
            "median_ms": statistics.median(values) if values else None,
            "max_ms": max(values) if values else None}


def summarize(run_dir: Path) -> Path:
    """정상 점수만 평균내고 N/A·실패·미검토를 분리해 10행 요약을 생성."""
    run_dir = Path(run_dir).resolve()
    records = load_records(run_dir)
    metrics = {}
    for name in METRICS:
        values = [r["ragas"][name]["score"] for r in records if r["ragas"][name]["evaluation_status"] == "completed"]
        metrics[name] = {"mean": statistics.mean(values) if values else None, "valid_count": len(values),
                         "status_counts": dict(Counter(r["ragas"][name]["evaluation_status"] for r in records))}
    rag_calls = [call for r in records for call in r["rag"]["api_calls"]]
    eval_calls = [call for r in records for call in r["api_calls"]]
    summary = {
        "eval_run_id": records[0]["eval_run_id"], "rag_run_id": records[0]["rag_run_id"],
        "question_count": len(records), "rag_status_counts": dict(Counter(r["rag"]["execution_status"] for r in records)),
        "metrics": metrics,
        "rules": dict(Counter(rule["verdict"] or rule["evaluation_status"] for r in records for rule in r["rules"])),
        "judge_status_counts": dict(Counter(r["judge"]["evaluation_status"] for r in records)),
        "judge_overall": dict(Counter(r["derived"]["judge_overall"] or "unavailable" for r in records)),
        "human_final": dict(Counter(r["derived"]["human_final"] or "not_reviewed" for r in records)),
        "disagreement_count": sum(len(r["human_reviews"][-1]["disagreements"]) for r in records if r["human_reviews"]),
        "rag_cost": aggregate_cost(rag_calls), "evaluation_cost": aggregate_cost(eval_calls),
        "cost_note": "RAG는 참조 실행 비용이며 이번 평가 비용에 재합산하지 않음. 사전 색인 비용 제외.",
        "rag_time": time_stats([r["rag"]["elapsed_ms"] for r in records if r["rag"]["execution_status"] == "completed" and r["rag"]["elapsed_ms"] is not None]),
        "evaluation_time": time_stats([r["derived"]["eval_elapsed_ms"] for r in records if r["derived"]["eval_elapsed_ms"] is not None and r["judge"]["evaluation_status"] == "completed" and all(m["evaluation_status"] in ("completed", "not_applicable") for m in r["ragas"].values())]),
        "performed_metric_counts": {r["question_id"]: r["derived"]["performed_metric_count"] for r in records},
    }
    write_json(run_dir / "summary.json", summary)
    with (run_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            row = {k: record[k] for k in ("eval_run_id", "rag_run_id", "question_id")}
            row.update({k: record["golden"][k] for k in ("topic", "expected_answerability")})
            row["rag_execution_status"] = record["rag"]["execution_status"]
            row["rag_error_summary"] = "; ".join(stage["error"]["message"] for stage in (record["rag"]["retrieval"], record["rag"]["generation"]) if stage["error"])
            for name, result in record["ragas"].items():
                row[name], row[f"{name}_status"] = result["score"], result["evaluation_status"]
            row["metric_notes"] = "; ".join(f"{k}: {v['reason']}" for k, v in record["ragas"].items() if v["evaluation_status"] != "completed")
            row["rules_summary"] = json.dumps(record["derived"]["rule_counts"], ensure_ascii=False)
            row["judge_status"] = record["judge"]["evaluation_status"]
            judge = record["judge"]["result"]
            row["observed_behavior"] = judge["behavior"]["observed"] if judge else None
            latest = record["human_reviews"][-1] if record["human_reviews"] else None
            for name, column in zip(DIMENSIONS, ("accuracy", "citation", "scope")):
                row[f"judge_{column}"] = judge[name]["verdict"] if judge else None
                row[f"human_{column}"] = latest[name]["verdict"] if latest else None
            row["human_review_status"] = "completed" if latest else "not_reviewed"
            row.update({k: v for k, v in record["derived"].items() if k in CSV_COLUMNS})
            row["detail_json_path"] = str(run_dir / f"{record['question_id']}.json")
            writer.writerow(row)
    return run_dir / "summary.csv"
