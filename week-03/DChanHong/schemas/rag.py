"""질문별 검색·생성 결과와 RAG 실행 설정을 보존하는 모델."""

from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from .common import (
    CodeInfo, Count, DatasetRef, ErrorInfo, FileRef, ModelConfig, NonEmpty,
    NonNegative, Number, Position, PricingSnapshot, PromptSnapshot, StrictModel,
    Timestamp, _validate_stage,
)
from .golden import GoldenDocument


class RetrievalHit(StrictModel):
    """반환 순서와 당시 원문 메타데이터를 보존하는 검색 청크."""

    label: NonEmpty
    chunk_id: NonEmpty
    returned_order: Position
    rank_in_revision: Position
    distance: Number
    metadata: dict[str, Any]
    text: str


class RetrievalResult(StrictModel):
    """질문 임베딩부터 검색 컨텍스트 구성까지의 결과."""

    execution_status: Literal["pending", "running", "completed", "skipped", "error"]
    results: list[RetrievalHit]
    elapsed_ms: NonNegative | None
    error: ErrorInfo | None

    @model_validator(mode="after")
    def validate_status(self) -> "RetrievalResult":
        """검색 오류·미실행 상태와 측정값의 조합을 확인."""
        _validate_stage(self.execution_status, self.error, self.elapsed_ms)
        return self


class GenerationResult(StrictModel):
    """실제로 보낸 생성 입력과 수신한 답변·원본 API 응답."""

    execution_status: Literal[
        "pending", "running", "completed", "needs_check", "skipped", "error"
    ]
    instructions: str | None
    input: str | None
    contexts: list[RetrievalHit]
    response: str | None
    api_response: dict[str, Any] | None
    elapsed_ms: NonNegative | None
    error: ErrorInfo | None

    @model_validator(mode="after")
    def validate_status(self) -> "GenerationResult":
        """빈 답변을 정상 완료로 저장하지 않고 원래 입력을 요구."""
        _validate_stage(self.execution_status, self.error, self.elapsed_ms)
        if self.execution_status == "completed":
            if self.response is None or not self.response.strip():
                raise ValueError("빈 생성 답변은 completed가 될 수 없습니다.")
            if self.instructions is None or self.input is None or self.api_response is None:
                raise ValueError("정상 생성 기록에는 실제 요청과 API 응답이 필요합니다.")
        return self


class RagRecord(StrictModel):
    """한 질문의 검색·실제 생성 요청·답변과 호출 기록."""

    schema_version: Literal["rag_record_v1"]
    rag_run_id: NonEmpty
    question_id: Annotated[str, Field(pattern=r"^Q(?:0[1-9]|10)$")]
    question: NonEmpty
    execution_status: Literal["pending", "running", "completed", "needs_check", "error"]
    stage: NonEmpty
    retrieval: RetrievalResult
    generation: GenerationResult
    api_calls: list[dict[str, Any]]
    started_at: Timestamp | None
    finished_at: Timestamp | None
    elapsed_ms: NonNegative | None

    @model_validator(mode="after")
    def validate_record(self) -> "RagRecord":
        """종료 시각·완료 상태와 반환 순서의 일관성을 확인."""
        if self.execution_status == "pending":
            if any(value is not None for value in (self.started_at, self.finished_at, self.elapsed_ms)):
                raise ValueError("실행 전 시간 필드는 모두 null입니다.")
        elif self.execution_status == "running":
            if self.started_at is None or self.finished_at is not None or self.elapsed_ms is not None:
                raise ValueError("실행 중에는 시작 시각만 존재해야 합니다.")
        elif self.started_at is None or self.finished_at is None:
            raise ValueError("종료 기록에는 시작·종료 시각이 필요합니다.")
        if self.started_at is not None and self.finished_at is not None:
            if self.finished_at < self.started_at:
                raise ValueError("종료 시각은 시작 시각보다 빠를 수 없습니다.")
        if self.execution_status == "completed" and (
            self.retrieval.execution_status != "completed"
            or self.generation.execution_status != "completed"
        ):
            raise ValueError("RAG 정상 완료에는 검색·생성 완료가 필요합니다.")
        for hits in (self.retrieval.results, self.generation.contexts):
            if [hit.returned_order for hit in hits] != list(range(1, len(hits) + 1)):
                raise ValueError("검색 결과 순위는 실제 반환 순서대로 1부터 이어져야 합니다.")
            if len({hit.label for hit in hits}) != len(hits):
                raise ValueError("한 컨텍스트 목록의 E 라벨은 중복될 수 없습니다.")
        return self


class ChunkingConfig(FileRef):
    """기존 청크 파일과 크기·중첩·분할 방식."""

    max_chars: Position
    overlap_chars: Count
    split_strategy: NonEmpty


class RetrievalConfig(StrictModel):
    """정정 구분별 개수와 결합 순서를 포함한 검색 설정."""

    collection: NonEmpty
    distance_metric: Literal["cosine"]
    top_k_per_revision: Position
    revision_order: list[Literal["before", "after"]]
    index_path: NonEmpty


class RagModels(StrictModel):
    """검색 임베딩과 답변 생성 모델을 구분한 설정."""

    embedding: ModelConfig
    generation: ModelConfig


class RagManifest(StrictModel):
    """한 RAG 실행의 입력·검색·생성 설정과 환경 기록."""

    schema_version: Literal["rag_manifest_v1"]
    rag_run_id: NonEmpty
    dataset: DatasetRef
    question_ids: list[Annotated[str, Field(pattern=r"^Q(?:0[1-9]|10)$")]]
    started_at: Timestamp
    finished_at: Timestamp | None
    code: CodeInfo
    documents: list[GoldenDocument]
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    models: RagModels
    prompt: PromptSnapshot
    pricing: PricingSnapshot

    @model_validator(mode="after")
    def validate_manifest(self) -> "RagManifest":
        """실행 목록의 중복과 시간 역전을 검증."""
        if len(self.question_ids) != len(set(self.question_ids)) or not self.question_ids:
            raise ValueError("실행 질문 목록은 비어 있거나 중복될 수 없습니다.")
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("실행 종료 시각이 시작 시각보다 빠릅니다.")
        return self
