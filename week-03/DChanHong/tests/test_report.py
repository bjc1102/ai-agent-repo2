"""합성 실행만으로 10행 요약과 사람 검토·비용 분리를 확인한다."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from config import DATASET_PATH
from evaluation.runner import pending_evaluation
from rag.pipeline import pending_record
from reporting.summary import derive, load_records, save_evaluation, summarize
from reporting.human_review import import_review, review_template
from schemas import GoldenDataset, HumanReview


def synthetic_rag(question):
    """정식 결과와 분리할 정상 완료 합성 RAG 기록을 구성."""
    record = pending_record("rag_offline_test", question.question_id, question.question)
    record.update(
        execution_status="completed", stage="completed",
        started_at="2026-11-14T00:00:00+09:00",
        finished_at="2026-11-14T00:00:01+09:00", elapsed_ms=1000,
        api_calls=[{"call_id": f"{question.question_id}.rag", "cost_usd": 1.0, "cost_complete": True}],
    )
    record["retrieval"].update(execution_status="completed", elapsed_ms=100)
    record["generation"].update(
        execution_status="completed", instructions="오프라인 테스트 지침",
        input=json.dumps({"question": question.question, "evidence": []}, ensure_ascii=False),
        response="오프라인 테스트용 합성 답변", api_response={"status": "completed"}, elapsed_ms=900,
    )
    return record


def completed_metric(score):
    """평균의 분모 검증에 쓸 점수와 합성 근거를 생성."""
    return {
        "evaluation_status": "completed", "score": score,
        "reason_code": "offline", "reason": "오프라인 합성 판정",
        "inputs": {"user_input": "합성 질문"}, "details": {"source": "offline_test"},
        "call_ids": [], "elapsed_ms": 10, "error": None,
    }


class ReportTests(unittest.TestCase):
    """실제 사람 판단을 만들지 않고 저장·집계 경계를 검증."""

    def setUp(self):
        """임시 폴더에만 10개 합성 평가 레코드를 만든다."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run_dir = Path(self.temporary.name) / "eval_offline_test"
        self.run_dir.mkdir()
        self.dataset = GoldenDataset.model_validate_json(DATASET_PATH.read_text(encoding="utf-8"))
        self.records = []
        for question in self.dataset.questions:
            record = pending_evaluation("eval_offline_test", question, synthetic_rag(question))
            record["api_calls"] = [{"call_id": f"{question.question_id}.eval", "cost_usd": 0.1, "cost_complete": True}]
            record["derived"] = derive(record, 20)
            self.records.append(record)
            self.save(record)

    def save(self, record):
        """실제 운영 저장 함수로 합성 레코드의 스키마를 검사."""
        save_evaluation(self.run_dir / f"{record['question_id']}.json", record)

    def test_pending_ten_questions_remain_in_summary_without_human_verdict(self):
        """대기 문항도 10행으로 남으며 사람이 검토하기 전 판정은 null."""
        csv_path = summarize(self.run_dir)
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        summary = json.loads((self.run_dir / "summary.json").read_text())
        self.assertEqual(len(rows), 10)
        self.assertEqual({row["question_id"] for row in rows}, {f"Q{i:02d}" for i in range(1, 11)})
        self.assertTrue(all(row["human_final"] == "" for row in rows))
        self.assertTrue(all(row["context_precision"] == "" for row in rows))
        self.assertEqual(summary["human_final"], {"not_reviewed": 10})
        self.assertIsNone(summary["metrics"]["context_precision"]["mean"])
        self.assertEqual(summary["metrics"]["context_precision"]["valid_count"], 0)
        self.assertEqual(summary["rag_cost"]["cost_usd"], 10.0)
        self.assertAlmostEqual(summary["evaluation_cost"]["cost_usd"], 1.0)

    def test_na_and_error_are_excluded_from_average_and_unknown_cost_retained(self):
        """적용 불가·오류를 0점으로 포함하지 않으며 미확보 비용을 보존."""
        self.records[0]["ragas"]["context_precision"] = completed_metric(0.2)
        self.records[1]["ragas"]["context_precision"] = completed_metric(0.8)
        for index in (6, 7):
            result = self.records[index]["ragas"]["context_precision"]
            result.update(evaluation_status="not_applicable", reason_code="no_positive_reference",
                          reason="답변 불가 문항", elapsed_ms=0)
        failed = self.records[2]
        failed["ragas"]["context_precision"].update(
            evaluation_status="error", reason_code="offline_failure", reason="합성 평가 오류",
            elapsed_ms=15, error={"code": "test_error", "message": "합성 오류"},
        )
        failed["api_calls"] = [{"call_id": "Q03.unknown", "cost_usd": None, "cost_complete": False}]
        for record in self.records:
            record["derived"] = derive(record, 20)
            self.save(record)
        summarize(self.run_dir)
        summary = json.loads((self.run_dir / "summary.json").read_text())
        precision = summary["metrics"]["context_precision"]
        self.assertEqual(precision["valid_count"], 2)
        self.assertAlmostEqual(precision["mean"], 0.5)
        self.assertEqual(precision["status_counts"], {"completed": 2, "error": 1, "pending": 5, "not_applicable": 2})
        self.assertFalse(summary["evaluation_cost"]["cost_complete"])
        self.assertIsNone(summary["evaluation_cost"]["cost_usd"])
        self.assertAlmostEqual(summary["evaluation_cost"]["known_cost_usd"], 0.9)
        self.assertEqual(summary["rag_cost"]["cost_usd"], 10)

    def test_reevaluation_cost_does_not_add_prior_rag_or_evaluation_calls(self):
        """같은 답변의 새 평가 비용에는 그 실행에서 발생한 호출만 포함."""
        question = self.dataset.questions[0]
        first = self.records[0]
        second = pending_evaluation("eval_new_test", question, first["rag"])
        second["api_calls"] = [{"call_id": "new_eval", "cost_usd": 0.02, "cost_complete": True}]
        second["derived"] = derive(second, 30)
        self.assertEqual(second["derived"]["rag_cost_usd"], 1)
        self.assertEqual(second["derived"]["eval_cost_usd"], 0.02)
        self.assertEqual(first["derived"]["eval_cost_usd"], 0.1)

    def test_human_review_is_blank_then_append_only(self):
        """미완성 양식은 판정이 되지 않고 완료 검토만 새 이력으로 추가."""
        template_path = Path(self.temporary.name) / "review.json"
        review_template(self.run_dir, "Q01", template_path)
        template = json.loads(template_path.read_text())
        with self.assertRaises(ValidationError):
            HumanReview.model_validate_json(template_path.read_text())
        with self.assertRaises(FileExistsError):
            review_template(self.run_dir, "Q01", template_path)
        self.assertIsNone(load_records(self.run_dir)[0]["derived"]["human_final"])
        automatic_before = copy.deepcopy(self.records[0]["ragas"])
        source = self.records[0]["golden"]["golden_sources"][0]
        review = {
            **template, "review_id": "review_offline_first", "reviewer": "offline_test",
            "reviewed_at": "2026-11-14T01:00:00+09:00",
            **{name: {"verdict": "pass", "reason": "합성 검토 판정", "basis_refs": [source["source_id"]]}
               for name in ("accuracy", "citation_appropriateness", "answer_scope")},
            "source_checks": [{"document_id": source["document_id"], "pdf_page": source["pdf_page"],
                               "excerpt": "오프라인 합성 검토용 구절"}],
        }
        template_path.write_text(json.dumps(review, ensure_ascii=False))
        import_review(self.run_dir, "Q01", template_path)
        first = load_records(self.run_dir)[0]
        self.assertEqual(first["derived"]["human_final"], "pass")
        self.assertEqual(first["ragas"], automatic_before)
        with self.assertRaises(ValueError):
            import_review(self.run_dir, "Q01", template_path)
        review["review_id"] = "review_offline_second"
        review["reviewed_at"] = "2026-11-14T01:01:00+09:00"
        review["accuracy"]["verdict"] = "fail"
        template_path.write_text(json.dumps(review, ensure_ascii=False))
        import_review(self.run_dir, "Q01", template_path)
        second = load_records(self.run_dir)[0]
        self.assertEqual(len(second["human_reviews"]), 2)
        self.assertEqual(second["human_reviews"][0], first["human_reviews"][0])
        self.assertEqual(second["derived"]["human_final"], "fail")
        self.assertEqual(second["ragas"], automatic_before)


if __name__ == "__main__":
    unittest.main()
