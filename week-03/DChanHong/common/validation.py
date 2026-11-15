"""골든·원문·설치 환경과 실행 전 준비 상태를 검증."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

from config import BASE_DIR, DATASET_PATH, CHUNKS_PATH, EXPECTED_VERSIONS
from schemas import GoldenDataset
from common.utils import sha256_file
from prompts.generation import GENERATION_PROMPT
from prompts.judge import JUDGE_PROMPT



def load_dataset(path: Path = DATASET_PATH) -> GoldenDataset:
    """골든 JSON의 타입·문항·참조와 원문 해시를 검증하여 반환."""
    dataset = GoldenDataset.model_validate_json(Path(path).read_text(encoding="utf-8"))
    for document in dataset.documents:
        source = Path(document.path)
        if not source.is_file():
            # 다른 기기에서도 파일명과 해시로 같은 원문을 찾습니다.
            source = BASE_DIR / "docs/01-document" / source.name
        if not source.is_file() or sha256_file(source) != document.sha256:
            raise ValueError(f"원문 파일 또는 SHA-256 불일치: {document.document_id}")
    return dataset


def environment_info() -> dict:
    """비밀 환경 변수 없이 코드 버전·작업 상태·설치 버전을 기록."""
    def git(*args: str) -> str:
        """읽기 전용 Git 정보만 조회."""
        result = subprocess.run(["git", *args], cwd=BASE_DIR, text=True,
                                capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""

    packages = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    return {"git_commit": git("rev-parse", "HEAD") or None,
            "git_dirty": bool(git("status", "--porcelain", "--", ".")),
            "python": platform.python_version(), "packages": packages}


def validate_versions() -> dict:
    """설치된 주요 패키지가 합의한 고정 버전인지 확인."""
    actual = environment_info()
    for package, version in EXPECTED_VERSIONS.items():
        if actual["packages"][package] != version:
            raise ValueError(f"{package} 버전 불일치: {version} 필요")
    return actual


def validate() -> None:
    """키·API 없이 데이터·색인·프롬프트·모듈의 준비 상태를 검증."""
    from rag.index import validate_index
    from evaluation.ragas_metrics import (
        AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness,
    )
    from schemas import JudgeResult

    dataset = load_dataset()
    environment = validate_versions()
    collection = validate_index()
    provenance = json.loads((BASE_DIR / "data/rag/baseline.json").read_text(encoding="utf-8"))
    for key in ("chunks", "sections"):
        asset = provenance[key]
        if sha256_file(BASE_DIR / asset["path"]) != asset["sha256"]:
            raise ValueError(f"2주차 기준 자료와 해시가 다릅니다: {key}")
    # Python 파일의 코드가 아닌 실제 전달 문자열을 기준 자료와 대조합니다.
    generation_hash = hashlib.sha256(GENERATION_PROMPT.encode("utf-8")).hexdigest()
    judge_hash = hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest()
    if not GENERATION_PROMPT.strip() or generation_hash != provenance["generation_prompt"]["sha256"]:
        raise ValueError("생성 프롬프트가 2주차 기준 자료와 다릅니다.")
    if not JUDGE_PROMPT.strip():
        raise ValueError("Judge 프롬프트가 비어 있습니다.")
    specification = (BASE_DIR / "docs/00-plan/EVALUATION_SPEC.md").read_text(encoding="utf-8")
    expected_judge = specification.split("### 6.2", 1)[1].split("```text\n", 1)[1].split("```", 1)[0].strip()
    if JUDGE_PROMPT != expected_judge:
        raise ValueError("Judge 프롬프트가 확정 문서와 다릅니다.")
    JudgeResult.model_json_schema()
    print(json.dumps({"validation": "passed", "questions": len(dataset.questions),
        "indexed_chunks": collection.count(), "chunks_sha256": sha256_file(CHUNKS_PATH),
        "generation_prompt_sha256": generation_hash, "judge_prompt_sha256": judge_hash,
        "python": environment["python"], "packages": environment["packages"],
        "ragas_metrics": [m.__name__ for m in (ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy)],
        "api_calls": 0, "credentials_checked": False}, ensure_ascii=False, indent=2))
