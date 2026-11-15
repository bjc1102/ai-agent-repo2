"""저장 모델이 공유하는 자료형·검증 함수와 실행 환경 기록 구조."""

from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, StrictInt


class StrictModel(BaseModel):
    """정의하지 않은 키를 거부하는 모든 저장 모델의 공통 기반."""

    model_config = ConfigDict(extra="forbid")


Number = StrictInt | Annotated[
    float, Field(strict=True, allow_inf_nan=False)
]


NonEmpty = Annotated[str, Field(min_length=1)]


NonNegative = Annotated[Number, Field(ge=0)]


Count = Annotated[int, Field(ge=0, strict=True)]


Position = Annotated[int, Field(ge=1, strict=True)]


Timestamp = Annotated[AwareDatetime, Field(strict=True)]


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


EvaluationStatus = Literal["pending", "completed", "not_applicable", "skipped", "error"]


class ErrorInfo(StrictModel):
    """실행 오류의 식별 코드와 사람이 읽을 수 있는 설명."""

    code: NonEmpty
    message: NonEmpty


def _validate_stage(status: str, error: ErrorInfo | None, elapsed_ms: Number | None) -> None:
    """실행 단계 공통의 오류와 경과 시간 상태 규칙을 확인."""
    if (status == "error") != (error is not None):
        raise ValueError("error 상태에서만 오류 객체가 필요합니다.")
    if status in ("pending", "running") and elapsed_ms is not None:
        raise ValueError("미완료 단계의 경과 시간은 null입니다.")
    if status in ("not_applicable", "skipped") and elapsed_ms != 0:
        raise ValueError("실행하지 않은 확정 단계의 경과 시간은 0입니다.")


class FileRef(StrictModel):
    """입력 파일의 당시 위치와 내용 해시."""

    path: NonEmpty
    sha256: Sha256


class DatasetRef(FileRef):
    """사용한 골든 파일과 논리 데이터셋 식별자."""

    dataset_id: NonEmpty


class CodeInfo(StrictModel):
    """실제 실행 코드와 Python·패키지 환경의 식별 정보."""

    git_commit: NonEmpty | None
    git_dirty: StrictBool
    python: NonEmpty
    packages: dict[NonEmpty, NonEmpty]


class ModelConfig(StrictModel):
    """모델의 고정 이름과 실제 요청에 쓰는 설정 원본."""

    model: NonEmpty
    settings: dict[str, Any]


class PromptSnapshot(StrictModel):
    """실행 당시 프롬프트 전문과 버전·해시."""

    version: NonEmpty
    text: NonEmpty
    sha256: Sha256


class SchemaSnapshot(StrictModel):
    """API에 사용한 출력 스키마와 그 내용 해시."""

    version: NonEmpty
    json_schema: dict[str, Any]
    sha256: Sha256


class PriceRate(StrictModel):
    """100만 토큰 기준 USD 단가이며 비적용 항목은 null."""

    input: NonNegative
    cached_input: NonNegative | None
    output: NonNegative | None
    source: NonEmpty


class PricingSnapshot(StrictModel):
    """비용 계산에 사용한 단가의 확인 시각·출처·모델별 값."""

    checked_at: Timestamp
    currency: Literal["USD"]
    unit: Literal["per_million_tokens"]
    conditions: NonEmpty
    models: dict[NonEmpty, PriceRate]
