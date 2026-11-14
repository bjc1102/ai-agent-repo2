"""자동 평가·사람 검토·비용 요약과 평가 실행 설정 모델."""

from typing import Annotated, Any, Literal

from pydantic import Field, StrictBool, model_validator

from .common import (
    CodeInfo, Count, DatasetRef, ErrorInfo, EvaluationStatus, FileRef, ModelConfig,
    NonEmpty, NonNegative, Number, Position, PricingSnapshot, PromptSnapshot,
    SchemaSnapshot, StrictModel, Timestamp, _validate_stage,
)
from .golden import GoldenQuestion, RuleExpectation
from .judge import JudgeResult
from .rag import RagRecord


class MetricResult(StrictModel):
    """한 Ragas 지표의 실제 입력·점수·상세 근거·적용 상태."""

    evaluation_status: EvaluationStatus
    score: Annotated[Number, Field(ge=-1, le=1)] | None
    reason_code: NonEmpty | None
    reason: NonEmpty | None
    inputs: dict[str, Any] | None
    details: dict[str, Any] | None
    call_ids: list[NonEmpty]
    elapsed_ms: NonNegative | None
    error: ErrorInfo | None

    @model_validator(mode="after")
    def validate_result(self) -> "MetricResult":
        """적용 불가·오류를 점수로 치환하지 못하도록 검증."""
        _validate_stage(self.evaluation_status, self.error, self.elapsed_ms)
        if self.evaluation_status == "completed":
            if self.score is None or self.inputs is None:
                raise ValueError("완료한 지표에는 점수와 실제 입력이 필요합니다.")
        elif self.score is not None:
            raise ValueError("미완료 지표의 점수는 null입니다.")
        if self.evaluation_status in ("not_applicable", "skipped", "error"):
            if self.reason_code is None or self.reason is None:
                raise ValueError("적용 불가·건너뜀·오류에는 사유가 필요합니다.")
        return self


class RagasResults(StrictModel):
    """적용 여부에 관계없이 네 지표를 모두 포함하는 결과 묶음."""

    context_precision: MetricResult
    context_recall: MetricResult
    faithfulness: MetricResult
    answer_relevancy: MetricResult

    @model_validator(mode="after")
    def validate_score_ranges(self) -> "RagasResults":
        """코사인 지표 이외의 점수는 0부터 1까지로 검증."""
        for metric in (self.context_precision, self.context_recall, self.faithfulness):
            if metric.score is not None and metric.score < 0:
                raise ValueError("검색 지표와 Faithfulness는 음수일 수 없습니다.")
        return self


class RuleResult(StrictModel):
    """기계적으로 확정 가능한 규칙의 기대값·관찰값·판정."""

    rule_id: NonEmpty
    evaluation_status: EvaluationStatus
    verdict: Literal["pass", "fail", "undetermined"] | None
    expected: RuleExpectation
    observed: Any | None
    response_quotes: list[str]
    reason: NonEmpty | None
    error: ErrorInfo | None

    @model_validator(mode="after")
    def validate_result(self) -> "RuleResult":
        """추출 불확실성과 실행 오류를 서로 구분해 저장."""
        if (self.evaluation_status == "error") != (self.error is not None):
            raise ValueError("규칙의 error 상태에만 오류 객체가 필요합니다.")
        if (self.evaluation_status == "completed") != (self.verdict is not None):
            raise ValueError("완료한 규칙에만 판정이 존재해야 합니다.")
        if self.evaluation_status != "pending" and self.reason is None:
            raise ValueError("검사한 규칙에는 판단 또는 미실행 이유가 필요합니다.")
        if self.rule_id != self.expected.rule_id:
            raise ValueError("규칙 결과와 기대 조건의 ID가 다릅니다.")
        return self


class JudgeEvaluation(StrictModel):
    """Judge의 원래 출력과 출력 검증 후 확정한 결과."""

    evaluation_status: EvaluationStatus
    result: JudgeResult | None
    raw_response: str | None
    call_ids: list[NonEmpty]
    elapsed_ms: NonNegative | None
    error: ErrorInfo | None

    @model_validator(mode="after")
    def validate_result(self) -> "JudgeEvaluation":
        """출력 오류를 품질 실패로 바꾸지 않고 원문을 보존."""
        _validate_stage(self.evaluation_status, self.error, self.elapsed_ms)
        if (self.evaluation_status == "completed") != (self.result is not None):
            raise ValueError("완료된 Judge에만 검증한 result가 존재합니다.")
        if self.evaluation_status == "completed" and self.raw_response is None:
            raise ValueError("Judge 완료 기록에는 원래 출력 문자열이 필요합니다.")
        return self


class HumanAssessment(StrictModel):
    """사람이 원문과 답변을 대조한 한 항목의 완료 판정."""

    verdict: Literal["pass", "fail"]
    reason: NonEmpty
    basis_refs: list[NonEmpty]


class SourceCheck(StrictModel):
    """사람이 직접 대조한 문서 페이지와 해당 구절."""

    document_id: NonEmpty
    pdf_page: Position
    excerpt: NonEmpty


class Disagreement(StrictModel):
    """자동 평가와 사람 판정이 다른 항목의 비교와 이유."""

    component: NonEmpty
    item: NonEmpty
    automatic_value: Any
    human_value: Any
    reason: NonEmpty


class HumanReview(StrictModel):
    """덮어쓰지 않고 누적하는 한 번의 완료된 사람 검토."""

    review_id: NonEmpty
    reviewer: NonEmpty
    reviewed_at: Timestamp
    accuracy: HumanAssessment
    citation_appropriateness: HumanAssessment
    answer_scope: HumanAssessment
    source_checks: list[SourceCheck] = Field(min_length=1)
    disagreements: list[Disagreement]
    notes: str


class DerivedResult(StrictModel):
    """자동·사람 판정 및 중복 없이 합산한 비용·시간 요약."""

    behavior_matches_expected: StrictBool | None
    judge_overall: Literal["pass", "fail", "needs_review"] | None
    rule_counts: dict[str, Count]
    latest_human_review_id: NonEmpty | None
    human_final: Literal["pass", "fail"] | None
    disagreement_summary: str
    rag_cost_usd: NonNegative | None
    rag_known_cost_usd: NonNegative
    rag_cost_complete: StrictBool
    rag_elapsed_ms: NonNegative | None
    eval_cost_usd: NonNegative | None
    eval_known_cost_usd: NonNegative
    eval_cost_complete: StrictBool
    eval_elapsed_ms: NonNegative | None
    performed_metric_count: Count

    @model_validator(mode="after")
    def validate_completeness(self) -> "DerivedResult":
        """미확보 비용을 0으로 채우거나 사람 판정을 미리 만들지 못하게 검증."""
        for prefix in ("rag", "eval"):
            complete = getattr(self, f"{prefix}_cost_complete")
            total = getattr(self, f"{prefix}_cost_usd")
            known = getattr(self, f"{prefix}_known_cost_usd")
            if complete != (total is not None):
                raise ValueError("비용 집계가 불완전하면 총비용은 null이어야 합니다.")
            if total is not None and abs(total - known) > 1e-12:
                raise ValueError("완전한 총비용과 확인된 비용 합계가 다릅니다.")
        if (self.latest_human_review_id is None) != (self.human_final is None):
            raise ValueError("사람 최종 판정과 최신 검토 ID는 함께 존재해야 합니다.")
        return self


class EvaluationRecord(StrictModel):
    """하나의 고정 RAG 답변에 대한 골든·자동 평가·사람 검토 전체."""

    schema_version: Literal["evaluation_record_v1"]
    eval_run_id: NonEmpty
    rag_run_id: NonEmpty
    question_id: Annotated[str, Field(pattern=r"^Q(?:0[1-9]|10)$")]
    golden: GoldenQuestion
    rag: RagRecord
    ragas: RagasResults
    rules: list[RuleResult]
    judge: JudgeEvaluation
    human_reviews: list[HumanReview]
    api_calls: list[dict[str, Any]]
    derived: DerivedResult

    @model_validator(mode="after")
    def validate_references(self) -> "EvaluationRecord":
        """복사본의 질문·실행 ID와 검토 이력 참조를 확인."""
        if self.rag_run_id != self.rag.rag_run_id:
            raise ValueError("평가와 RAG 복사본의 실행 ID가 다릅니다.")
        if self.question_id != self.golden.question_id or self.question_id != self.rag.question_id:
            raise ValueError("평가·골든·RAG의 문항 ID가 다릅니다.")
        if self.golden.question != self.rag.question:
            raise ValueError("골든과 RAG에 기록한 실제 질문이 다릅니다.")
        review_ids = [review.review_id for review in self.human_reviews]
        if len(review_ids) != len(set(review_ids)):
            raise ValueError("사람 검토 이력 ID가 중복되었습니다.")
        if self.human_reviews:
            latest = self.human_reviews[-1]
            if self.derived.latest_human_review_id != latest.review_id:
                raise ValueError("최신 사람 검토 ID는 마지막 추가 이력을 가리켜야 합니다.")
            final = "pass" if all(
                getattr(latest, dimension).verdict == "pass"
                for dimension in ("accuracy", "citation_appropriateness", "answer_scope")
            ) else "fail"
            if self.derived.human_final != final:
                raise ValueError("사람 최종 판정은 세 항목이 모두 통과할 때만 pass입니다.")
        elif self.derived.latest_human_review_id is not None:
            raise ValueError("미검토 문항에는 사람 판정이 존재할 수 없습니다.")
        return self


class EvaluationModels(StrictModel):
    """Ragas·평가 임베딩·별도 Judge 모델의 설정."""

    ragas_llm: ModelConfig
    embedding: ModelConfig
    judge: ModelConfig


class EvaluationSettings(StrictModel):
    """프레임워크·규칙 버전과 지표에 실제 넘기는 설정."""

    ragas_version: NonEmpty
    metric_settings: dict[str, Any]
    rules_version: NonEmpty


class EvaluationManifest(StrictModel):
    """저장된 RAG 답변에 적용한 평가 환경·기준의 실행 기록."""

    schema_version: Literal["evaluation_manifest_v1"]
    eval_run_id: NonEmpty
    rag_run_id: NonEmpty
    started_at: Timestamp
    finished_at: Timestamp | None
    code: CodeInfo
    dataset: DatasetRef
    rag_inputs: list[FileRef]
    settings: EvaluationSettings
    models: EvaluationModels
    judge_prompt: PromptSnapshot
    judge_schema: SchemaSnapshot
    pricing: PricingSnapshot

    @model_validator(mode="after")
    def validate_manifest(self) -> "EvaluationManifest":
        """평가 실행의 시작·종료 시각 순서를 확인."""
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("평가 종료 시각이 시작 시각보다 빠릅니다.")
        return self
