import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parents[2]
input_path = BASE_DIR / "output" / "chunks.json"
db_path = BASE_DIR / "chroma_db"

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
BATCH_SIZE = 32
COLLECTION_NAME = "dongwon_20261030_v1"

load_dotenv(BASE_DIR / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY 설정을 확인해주세요.")

# 입력 파일을 읽고, 파일 변경 여부를 확인할 지문을 만듭니다.
input_bytes = input_path.read_bytes()
chunks = json.loads(input_bytes)
input_hash = hashlib.sha256(input_bytes).hexdigest()

expected = {chunk["id"]: chunk for chunk in chunks}

if not chunks or len(expected) != len(chunks):
    raise SystemExit("청크가 없거나 중복 ID가 있습니다.")

if any(not chunk["text"].strip() for chunk in chunks):
    raise SystemExit("본문이 비어 있는 청크가 있습니다.")

# 사용할 모델과 입력 파일 정보를 컬렉션에도 기록합니다.
settings = {
    "embedding_model": MODEL,
    "embedding_dimensions": DIMENSIONS,
    "chunks_sha256": input_hash,
    "distance_metric": "cosine",
}

db = chromadb.PersistentClient(path=str(db_path))
collection = db.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=None,
    configuration={"hnsw": {"space": "cosine"}},
    metadata=settings,
)

# 다른 입력이나 모델로 만든 저장소를 섞어 쓰지 않도록 확인합니다.
saved_settings = collection.metadata or {}
if any(saved_settings.get(key) != value for key, value in settings.items()):
    raise SystemExit(
        "기존 컬렉션의 입력 파일 또는 모델 설정이 다릅니다. "
        "COLLECTION_NAME을 새로운 이름으로 바꿔주세요."
    )

# 이미 저장된 청크는 API를 다시 호출하지 않습니다.
stored_ids = set(collection.get(include=[])["ids"])

if stored_ids - set(expected):
    raise SystemExit("컬렉션에 현재 입력 파일에 없는 ID가 있습니다.")

pending = [
    chunk for chunk in chunks
    if chunk["id"] not in stored_ids
]

print(f"전체 청크 수: {len(chunks)}")
print(f"이미 저장된 청크 수: {len(stored_ids)}")
print(f"이번에 처리할 청크 수: {len(pending)}", flush=True)

client = OpenAI(timeout=60.0, max_retries=2)
processed = len(stored_ids)
total_tokens = 0

for start in range(0, len(pending), BATCH_SIZE):
    batch = pending[start:start + BATCH_SIZE]

    response = client.embeddings.create(
        model=MODEL,
        dimensions=DIMENSIONS,
        input=[chunk["text"] for chunk in batch],
        encoding_format="float",
    )

    # 응답의 index를 기준으로 입력 청크와 순서를 맞춥니다.
    items = sorted(response.data, key=lambda item: item.index)

    if [item.index for item in items] != list(range(len(batch))):
        raise RuntimeError("임베딩 응답과 입력 청크의 개수가 맞지 않습니다.")

    embeddings = [item.embedding for item in items]

    if any(len(vector) != DIMENSIONS for vector in embeddings):
        raise RuntimeError("예상한 벡터 차원과 다릅니다.")

    # ID, 벡터, 본문, 근거 정보를 함께 저장합니다.
    collection.upsert(
        ids=[chunk["id"] for chunk in batch],
        embeddings=embeddings,
        documents=[chunk["text"] for chunk in batch],
        metadatas=[chunk["metadata"] for chunk in batch],
    )

    processed += len(batch)
    total_tokens += response.usage.total_tokens
    print(f"저장 진행: {processed}/{len(chunks)}", flush=True)

# 저장소에서 다시 읽어 입력 데이터와 대조합니다.
saved = collection.get(
    include=["documents", "metadatas", "embeddings"]
)

if (
    len(saved["ids"]) != len(chunks)
    or set(saved["ids"]) != set(expected)
):
    raise RuntimeError("저장된 ID 목록이 입력과 다릅니다.")

if saved["embeddings"] is None:
    raise RuntimeError("저장된 벡터를 읽지 못했습니다.")

for index, chunk_id in enumerate(saved["ids"]):
    original = expected[chunk_id]

    if saved["documents"][index] != original["text"]:
        raise RuntimeError(f"본문 불일치: {chunk_id}")

    if saved["metadatas"][index] != original["metadata"]:
        raise RuntimeError(f"메타데이터 불일치: {chunk_id}")

    if len(saved["embeddings"][index]) != DIMENSIONS:
        raise RuntimeError(f"벡터 차원 불일치: {chunk_id}")

counts = Counter(
    metadata["revision"] for metadata in saved["metadatas"]
)

print("\n===== 저장 결과 =====")
print(f"컬렉션: {collection.name}")
print(f"최종 저장 청크 수: {collection.count()}")
print(f"정정 전: {counts['before']}")
print(f"정정 후: {counts['after']}")
print(f"요약·안내: {counts['overview']}")
print(f"벡터 차원: {DIMENSIONS}")
print(f"이번 실행 사용 토큰 수: {total_tokens}")
print("본문·메타데이터·벡터 차원 검증 완료")