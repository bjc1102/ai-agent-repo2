"""실제 API 없이 골든 참조·상태·결측값 경계를 검증한다."""

import copy
import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from schemas import (
    DerivedResult,
    GenerationResult,
    GoldenDataset,
    JudgeEvaluation,
    JudgeResult,
    MetricResult,
    RagasResults,
)


def validate(model, value):
    """파일 저장 때와 같은 JSON 경로로 모델을 검증."""
    return model.model_validate_json(json.dumps(value, ensure_ascii=False))


def metric(status="not_applicable"):
    """적용 불가 지표를 위한 최소한의 합성 기록을 생성."""
    return {
        "evaluation_status": status,
        "score": None,
        "reason_code": "policy",
        "reason": "이 합성 질문은 지표 적용 대상이 아닙니다.",
        "inputs": None,
        "details": None,
        "call_ids": [],
        "elapsed_ms": 0,
        "error": None,
    }


class SchemaTests(unittest.TestCase):
    """잘못된 참조·가짜 점수·미확보 비용을 저장하지 못하게 확인."""

    @classmethod
    def setUpClass(cls):
        """검증 대상인 실제 골든 JSON을 읽어 둔다."""
        path = Path(__file__).resolve().parents[1] / "data/golden/golden_v1.json"
        cls.golden = json.loads(path.read_text(encoding="utf-8"))

    def test_golden_complete_and_unknown_reference_rejected(self):
        """실제 10문항은 허용하고 깨진 원문 참조는 거부."""
        dataset = validate(GoldenDataset, self.golden)
        self.assertEqual(len(dataset.questions), 10)
        broken = copy.deepcopy(self.golden)
        broken["questions"][0]["golden_criteria"][0]["source_refs"] = ["G999"]
        with self.assertRaises(ValidationError):
            validate(GoldenDataset, broken)

    def test_naive_timestamp_and_boolean_number_rejected(self):
        """시간대 없는 시각과 bool을 숫자로 저장하는 오류를 거부."""
        broken = copy.deepcopy(self.golden)
        broken["created_at"] = "2026-11-14T15:27:49"
        with self.assertRaises(ValidationError):
            validate(GoldenDataset, broken)
        broken = copy.deepcopy(self.golden)
        broken["questions"][0]["rule_expectations"][0]["expected_value"] = True
        with self.assertRaises(ValidationError):
            validate(GoldenDataset, broken)

    def test_empty_received_answer_is_preserved_as_needs_check(self):
        """실제 빈 응답은 보존하되 정상 완료로 둔갑시키지 않음."""
        record = {
            "execution_status": "needs_check",
            "instructions": "합성 지침",
            "input": '{"question":"합성 질문","evidence":[]}',
            "contexts": [],
            "response": "",
            "api_response": {"status": "completed", "output": []},
            "elapsed_ms": 15.0,
            "error": None,
        }
        self.assertEqual(validate(GenerationResult, record).response, "")
        record["execution_status"] = "completed"
        with self.assertRaises(ValidationError):
            validate(GenerationResult, record)

    def test_not_applicable_is_not_a_score(self):
        """N/A는 null 점수와 미실행 시간 0으로 구분."""
        record = metric()
        self.assertIsNone(validate(MetricResult, record).score)
        record["score"] = 0
        with self.assertRaises(ValidationError):
            validate(MetricResult, record)

    def test_metric_errors_keep_error_and_no_score(self):
        """평가 오류에는 오류 객체가 필요하며 점수는 존재하지 않음."""
        record = metric("error")
        record["error"] = {"code": "parse_failure", "message": "합성 형식 오류"}
        record["elapsed_ms"] = 10
        self.assertEqual(validate(MetricResult, record).evaluation_status, "error")
        record["score"] = 1
        with self.assertRaises(ValidationError):
            validate(MetricResult, record)

    def test_only_answer_relevancy_accepts_negative_score(self):
        """코사인 유사도 지표만 음수 점수를 허용."""
        record = {name: metric() for name in (
            "context_precision", "context_recall", "faithfulness", "answer_relevancy"
        )}
        measured = metric("completed")
        measured.update(score=-0.1, inputs={"response": "합성 답변"})
        record["answer_relevancy"] = measured
        self.assertEqual(validate(RagasResults, record).answer_relevancy.score, -0.1)
        record["context_precision"] = measured
        with self.assertRaises(ValidationError):
            validate(RagasResults, record)

    def test_judge_needs_review_is_completed(self):
        """판정 보류를 출력 검증 오류와 구분."""
        assessment = {
            "verdict": "needs_review", "reason": "원문 재확인 필요",
            "response_quotes": [], "basis_refs": [],
        }
        result = {
            "behavior": {"observed": "unclear", "reason": "합성 답변 불명확"},
            "accuracy": assessment, "citation_appropriateness": assessment,
            "answer_scope": assessment,
        }
        validate(JudgeResult, result)
        record = {
            "evaluation_status": "completed", "result": result,
            "raw_response": json.dumps(result), "call_ids": [],
            "elapsed_ms": 10, "error": None,
        }
        self.assertEqual(validate(JudgeEvaluation, record).result.accuracy.verdict, "needs_review")
        record["evaluation_status"] = "error"
        record["error"] = {"code": "parse_failure", "message": "합성 오류"}
        with self.assertRaises(ValidationError):
            validate(JudgeEvaluation, record)

    def test_unknown_cost_is_not_zero(self):
        """확인된 부분 합계와 미확보 총비용을 구별."""
        record = {
            "behavior_matches_expected": None, "judge_overall": None,
            "rule_counts": {}, "latest_human_review_id": None, "human_final": None,
            "disagreement_summary": "", "rag_cost_usd": None,
            "rag_known_cost_usd": 0.01, "rag_cost_complete": False,
            "rag_elapsed_ms": 10, "eval_cost_usd": 0,
            "eval_known_cost_usd": 0, "eval_cost_complete": True,
            "eval_elapsed_ms": 0, "performed_metric_count": 0,
        }
        self.assertIsNone(validate(DerivedResult, record).rag_cost_usd)
        record["rag_cost_usd"] = 0
        with self.assertRaises(ValidationError):
            validate(DerivedResult, record)


if __name__ == "__main__":
    unittest.main()
