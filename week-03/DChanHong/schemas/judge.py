"""Judge API가 반환하는 답변 방식과 세 항목의 판정 모델."""

from typing import Literal

from .common import StrictModel


class Behavior(StrictModel):
    """Judge가 관찰한 실제 답변 방식과 판단 이유."""

    observed: Literal["answer", "abstain", "partial_answer", "unclear"]
    reason: str


class Assessment(StrictModel):
    """Judge의 한 평가 항목 판정과 실제 답변·근거 참조."""

    verdict: Literal["pass", "fail", "needs_review"]
    reason: str
    response_quotes: list[str]
    basis_refs: list[str]


class JudgeResult(StrictModel):
    """확정한 judge_result_v1 출력 키와 enum을 보존하는 API 모델."""

    behavior: Behavior
    accuracy: Assessment
    citation_appropriateness: Assessment
    answer_scope: Assessment
