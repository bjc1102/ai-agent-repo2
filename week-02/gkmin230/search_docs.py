
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# 기존 Chroma DB 불러오기
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_collection(name="guardduty_docs")

# 검색할 질문
question = "SSHBruteForce 탐지는 무엇을 의미하나요?"

# 질문도 같은 임베딩 모델로 벡터 변환
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=question
)

question_vector = response.data[0].embedding

# 유사도가 높은 문서 3개 검색
results = collection.query(
    query_embeddings=[question_vector],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

print(f"\n질문: {question}")

for i, (document, metadata, distance) in enumerate(
    zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ),
    start=1
):
    print(f"\n===== 검색 결과 {i} =====")
    print(f"거리 점수: {distance:.4f}")
    print(f"출처: {metadata['source']}")
    print(f"청크 번호: {metadata['chunk_index']}")
    print(f"내용:\n{document}")
