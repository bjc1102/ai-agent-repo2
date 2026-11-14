"""평가 질문·골든 답변·원문 근거와 문항별 적용 정책 모델."""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from .common import Number, StrictModel


class GoldenDocument(StrictModel):
    """평가에 사용하는 원본 문서의 식별 정보와 파일 해시."""

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class GoldenCriterion(StrictModel):
    """문항의 정확성·인용·답변 범위를 판단하는 골든 기준."""

    criterion_id: str = Field(min_length=1)
    dimension: Literal[
        "accuracy",
        "citation_appropriateness",
        "answer_scope",
    ]
    kind: Literal["required", "forbidden", "allowed"]
    description: str = Field(min_length=1)
    source_refs: list[str]


class GoldenSource(StrictModel):
    """골든 답변의 근거가 되는 원문 발췌와 문서 내 위치."""

    source_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    pdf_page: int = Field(ge=1, strict=True)
    printed_page: str | None
    revision: Literal["before", "after"] | None
    section: str | None
    excerpt: str = Field(min_length=1)
    role: Literal["answer_evidence", "related_context"]


class SourceLocation(StrictModel):
    """정보의 존재 여부를 확인한 문서 위치."""

    document_id: str = Field(min_length=1)
    pdf_page: int = Field(ge=1, strict=True)
    section: str | None


class AbsenceCheck(StrictModel):
    """답변 불가로 판단하기 전에 무엇을 어디서 확인했는지 기록."""

    target: str = Field(min_length=1)
    search_terms: list[str] = Field(min_length=1)
    reviewed_locations: list[SourceLocation] = Field(min_length=1)
    conclusion: str = Field(min_length=1)


class NumericRange(StrictModel):
    """희망 공모가처럼 최솟값과 최댓값이 있는 수치 범위."""

    min: Number
    max: Number

    @model_validator(mode="after")
    def validate_bounds(self) -> "NumericRange":
        """최솟값이 최댓값보다 큰 잘못된 범위를 거부."""
        if self.min > self.max:
            raise ValueError("min은 max보다 클 수 없습니다.")
        return self


class RuleExpectation(StrictModel):
    """수치·범위·계산·인용 라벨 검사에 사용할 기대 조건."""

    rule_id: str = Field(min_length=1)
    kind: Literal[
        "numeric", "numeric_range", "calculation", "citation_labels"
    ]
    target: str = Field(min_length=1)
    revision: Literal["before", "after"] | None
    period: str | None
    expected_value: Number | NumericRange | None
    unit: str | None = Field(min_length=1)
    formula: str | None = Field(min_length=1)
    rounding_digits: int | None = Field(ge=0, strict=True)
    tolerance: Annotated[Number, Field(ge=0)] | None

    @model_validator(mode="after")
    def validate_rule(self) -> "RuleExpectation":
        """규칙 종류와 기대값·계산식의 조합을 확인."""
        if self.kind in ("numeric", "calculation"):
            if not isinstance(self.expected_value, (int, float)):
                raise ValueError("수치·계산 규칙에는 숫자 기대값이 필요합니다.")

        elif self.kind == "numeric_range":
            if not isinstance(self.expected_value, NumericRange):
                raise ValueError("범위 규칙에는 min과 max가 필요합니다.")

        elif self.kind == "citation_labels":
            numeric_fields = (
                self.expected_value,
                self.unit,
                self.rounding_digits,
                self.tolerance,
            )
            if any(value is not None for value in numeric_fields):
                raise ValueError("인용 라벨 규칙의 수치 관련 필드는 null입니다.")

        if self.kind == "calculation":
            if self.formula is None:
                raise ValueError("계산 규칙에는 formula가 필요합니다.")
        elif self.formula is not None:
            raise ValueError("계산 규칙 외에는 formula를 null로 둡니다.")

        return self


class MetricPolicy(StrictModel):
    """한 평가 지표의 실행 여부·평가 범위·적용 이유."""

    mode: Literal["run", "conditional", "not_applicable"]
    scope: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class MetricPolicies(StrictModel):
    """문항별 Ragas 네 지표의 적용 정책."""

    context_precision: MetricPolicy
    context_recall: MetricPolicy
    faithfulness: MetricPolicy
    answer_relevancy: MetricPolicy


class GoldenQuestion(StrictModel):
    """평가 질문 하나의 정답·근거·규칙·지표 적용 정책."""

    question_id: str = Field(pattern=r"^Q(?:0[1-9]|10)$")
    topic: Literal["offering_price", "offering_terms", "fund_use"]
    question: str = Field(min_length=1)
    expected_answerability: Literal["answerable", "unanswerable", "partial"]
    golden_answer: str = Field(min_length=1)
    answerable_parts: list[Annotated[str, Field(min_length=1)]]
    unanswerable_parts: list[Annotated[str, Field(min_length=1)]]
    golden_criteria: list[GoldenCriterion]
    golden_sources: list[GoldenSource]
    absence_checks: list[AbsenceCheck]
    retrieval_reference: str | None = Field(min_length=1)
    rule_expectations: list[RuleExpectation]
    metric_policy: MetricPolicies

    @model_validator(mode="after")
    def validate_question_policy(self) -> "GoldenQuestion":
        """문항별 답변 가능 범위와 지표 정책이 합의 내용과 맞는지 확인."""
        number = int(self.question_id[1:])

        if number <= 6:
            expected = "answerable"
        elif number <= 8:
            expected = "unanswerable"
        else:
            expected = "partial"

        if self.expected_answerability != expected:
            raise ValueError(
                f"{self.question_id}의 expected_answerability는 {expected}입니다."
            )

        has_answerable_part = expected != "unanswerable"
        has_unanswerable_part = expected != "answerable"

        if bool(self.answerable_parts) != has_answerable_part:
            raise ValueError("문항 유형에 맞게 answerable_parts를 작성하세요.")

        if bool(self.unanswerable_parts) != has_unanswerable_part:
            raise ValueError("문항 유형에 맞게 unanswerable_parts를 작성하세요.")

        if bool(self.absence_checks) != has_unanswerable_part:
            raise ValueError("Q07~Q10에는 답변 불가 부분의 검토 기록이 필요합니다.")

        if (self.retrieval_reference is not None) != has_answerable_part:
            raise ValueError("retrieval_reference는 Q07·Q08에서만 null입니다.")

        retrieval_mode = "run" if has_answerable_part else "not_applicable"
        expected_modes = {
            "context_precision": retrieval_mode,
            "context_recall": retrieval_mode,
            "faithfulness": "conditional",
            "answer_relevancy": "run" if number <= 6 else "not_applicable",
        }

        for metric_name, expected_mode in expected_modes.items():
            policy = getattr(self.metric_policy, metric_name)
            if policy.mode != expected_mode:
                raise ValueError(
                    f"{self.question_id}의 {metric_name}.mode는 "
                    f"{expected_mode}입니다."
                )

        for items, key in (
            (self.golden_criteria, "criterion_id"),
            (self.golden_sources, "source_id"),
            (self.rule_expectations, "rule_id"),
        ):
            identifiers = [getattr(item, key) for item in items]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"{self.question_id}의 {key}가 중복되었습니다.")
        source_ids = {source.source_id for source in self.golden_sources}
        if any(
            reference not in source_ids
            for criterion in self.golden_criteria
            for reference in criterion.source_refs
        ):
            raise ValueError("골든 기준이 존재하지 않는 G 근거를 참조합니다.")

        return self


class GoldenDataset(StrictModel):
    """원본 문서 정보와 확정한 평가 질문 10개를 묶은 데이터셋."""

    schema_version: Literal["golden_dataset_v1"]
    dataset_id: str = Field(min_length=1)
    created_at: AwareDatetime = Field(strict=True)
    documents: list[GoldenDocument] = Field(min_length=1)
    questions: list[GoldenQuestion] = Field(min_length=10, max_length=10)

    @model_validator(mode="after")
    def validate_question_ids(self) -> "GoldenDataset":
        """Q01부터 Q10까지 누락·중복 없이 포함됐는지 확인."""
        expected_ids = {f"Q{number:02d}" for number in range(1, 11)}
        actual_ids = {question.question_id for question in self.questions}

        if actual_ids != expected_ids:
            raise ValueError("Q01~Q10이 각각 한 번씩 있어야 합니다.")

        document_ids = [document.document_id for document in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document_id가 중복되었습니다.")
        for question in self.questions:
            locations = list(question.golden_sources) + [
                location
                for check in question.absence_checks
                for location in check.reviewed_locations
            ]
            if any(location.document_id not in document_ids for location in locations):
                raise ValueError("근거가 존재하지 않는 문서를 참조합니다.")

        return self
