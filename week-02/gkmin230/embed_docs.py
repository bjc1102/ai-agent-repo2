
import json
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# API 키 불러오기
load_dotenv()
client = OpenAI()

# 기존에 만든 73개 청크 불러오기
chunks = json.loads(
    Path("data/chunks.json").read_text(encoding="utf-8")
)

print(f"불러온 청크: {len(chunks)}개")

# 청크를 숫자 벡터로 변환
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=[chunk["text"] for chunk in chunks]
)

embeddings = [item.embedding for item in response.data]

print(f"임베딩 생성 완료: {len(embeddings)}개")

# Chroma 벡터 DB 생성
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection(
    name="guardduty_docs"
)

# 벡터와 문서 저장
collection.upsert(
    ids=[chunk["id"] for chunk in chunks],
    embeddings=embeddings,
    documents=[chunk["text"] for chunk in chunks],
    metadatas=[
        {
            "source": chunk["source"],
            "url": chunk["url"],
            "chunk_index": chunk["chunk_index"]
        }
        for chunk in chunks
    ]
)

print("Chroma 저장 완료!")
print(f"저장된 청크 수: {collection.count()}개")
