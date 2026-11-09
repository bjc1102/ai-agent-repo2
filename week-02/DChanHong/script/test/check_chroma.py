from pathlib import Path

import chromadb


BASE_DIR = Path(__file__).resolve().parents[2]
db_path = BASE_DIR / "chroma_db"

# 지정한 폴더에 데이터를 저장하는 로컬 Chroma를 엽니다.
client = chromadb.PersistentClient(path=str(db_path))

# 컬렉션은 청크와 벡터를 모아두는 저장 단위입니다.
# 이번에는 빈 컬렉션을 만들고, 임베딩 모델은 나중에 연결합니다.
collection = client.get_or_create_collection(
    name="setup_check",
    embedding_function=None,
)

print(f"Chroma 버전: {client.get_version()}")
print(f"저장 위치: {db_path}")
print(f"컬렉션: {collection.name}")
print(f"저장된 청크 수: {collection.count()}")
print("Chroma 초기화 성공")