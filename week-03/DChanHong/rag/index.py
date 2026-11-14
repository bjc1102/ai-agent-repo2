"""기존 2주차 색인을 복사하고 청크·벡터의 일치 여부를 검증한다."""

import json
import shutil
import tempfile
from pathlib import Path

from common.utils import sha256_file
from config import (
    BASE_DIR,
    CHUNKS_PATH,
    COLLECTION_NAME,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    INDEX_PATH,
)


def validate_index(index_path: Path = INDEX_PATH):
    """색인의 설정·본문·메타데이터·벡터를 원래 청크와 대조한다."""
    import chromadb
    from chromadb.config import Settings

    index_path = Path(index_path)
    if not (index_path / "chroma.sqlite3").is_file():
        raise FileNotFoundError(
            f"색인이 없습니다: {index_path}. prepare-index 명령으로 기존 색인을 복사하세요."
        )
    chunks = json.loads(CHUNKS_PATH.read_bytes())
    expected = {chunk["id"]: chunk for chunk in chunks}
    if not chunks or len(expected) != len(chunks):
        raise ValueError("청크가 비어 있거나 ID가 중복됩니다.")
    client = chromadb.PersistentClient(
        path=str(index_path), settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(
        name=COLLECTION_NAME, embedding_function=None
    )
    required = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "distance_metric": "cosine",
        "chunks_sha256": sha256_file(CHUNKS_PATH),
    }
    if any((collection.metadata or {}).get(key) != value for key, value in required.items()):
        raise ValueError("색인의 모델·차원·거리·청크 해시가 기준과 다릅니다.")
    if (collection.configuration.get("hnsw") or {}).get("space") != "cosine":
        raise ValueError("색인의 실제 HNSW 거리 설정이 cosine이 아닙니다.")
    saved = collection.get(include=["documents", "metadatas", "embeddings"])
    if len(saved["ids"]) != len(chunks) or set(saved["ids"]) != set(expected):
        raise ValueError("색인에 저장된 청크 ID가 입력과 다릅니다.")
    if saved["embeddings"] is None:
        raise ValueError("색인에서 벡터를 읽지 못했습니다.")
    for position, chunk_id in enumerate(saved["ids"]):
        original = expected[chunk_id]
        if saved["documents"][position] != original["text"]:
            raise ValueError(f"청크 본문이 다릅니다: {chunk_id}")
        if saved["metadatas"][position] != original["metadata"]:
            raise ValueError(f"청크 메타데이터가 다릅니다: {chunk_id}")
        if len(saved["embeddings"][position]) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"청크 벡터 차원이 다릅니다: {chunk_id}")
    return collection


def prepare_index(source: Path | None = None) -> Path:
    """임베딩 API를 호출하지 않고 기존 색인을 독립된 작업 폴더로 복사한다."""
    source = Path(source) if source is not None else BASE_DIR.parents[1] / "week-02/DChanHong/chroma_db"
    if INDEX_PATH.exists():
        validate_index(INDEX_PATH)
        return INDEX_PATH
    if not (source / "chroma.sqlite3").is_file():
        raise FileNotFoundError(f"복사할 기존 색인이 없습니다: {source}")
    if source.resolve() == INDEX_PATH.resolve():
        raise ValueError("원본과 복사 대상은 달라야 합니다.")
    before = {
        path.relative_to(source): sha256_file(path)
        for path in source.rglob("*") if path.is_file()
    }
    with tempfile.TemporaryDirectory(prefix="index-copy-", dir=BASE_DIR) as temporary:
        copied = Path(temporary) / "chroma_db"
        shutil.copytree(source, copied)
        for relative, digest in before.items():
            if sha256_file(source / relative) != digest or sha256_file(copied / relative) != digest:
                raise RuntimeError("복사 중 색인이 변경됐습니다. 실행 중인 원본 색인 작업을 확인하세요.")
        copied.rename(INDEX_PATH)
    validate_index(INDEX_PATH)
    return INDEX_PATH
