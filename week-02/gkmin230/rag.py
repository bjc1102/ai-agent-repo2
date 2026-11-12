
import json
from datetime import datetime
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# 기존 Chroma DB 연결
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_collection(name="guardduty_docs")

EMBEDDING_MODEL = "text-embedding-3-small"
ANSWER_MODEL = "gpt-4o-mini"
TOP_K = 3

# 실험 질문 3개
questions = [
    "SSHBruteForce 탐지는 무엇을 의미하나요?",
    "Portscan 탐지는 무엇을 의미하나요?",
    "어제 우리 회사에서 발생한 GuardDuty 탐지 건수는?"
]

experiment_results = []

for number, question in enumerate(questions, start=1):

    print(f"\n===== 실험 {number} =====")
    print(f"질문: {question}")

    # 1. 질문 임베딩
    embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question
    )

    question_vector = embedding_response.data[0].embedding

    # 2. Chroma 문서 검색
    search_results = collection.query(
        query_embeddings=[question_vector],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_docs = []
    contexts = []

    for i, (document, metadata, distance) in enumerate(
        zip(
            search_results["documents"][0],
            search_results["metadatas"][0],
            search_results["distances"][0]
        ),
        start=1
    ):
        retrieved_docs.append({
            "rank": i,
            "chunk_index": metadata["chunk_index"],
            "source": metadata["source"],
            "url": metadata["url"],
            "distance": round(float(distance), 4),
            "text": document
        })

        contexts.append(f"[문서 {i}]\n{document}")

        print(
            f"검색 {i}: 청크 {metadata['chunk_index']}, "
            f"거리 {distance:.4f}"
        )

    context_text = "\n\n".join(contexts)

    # 3. GPT 답변 생성
    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "너는 AWS GuardDuty 보안 분석 도우미다. "
                    "반드시 제공된 참고 문서에 근거하여 한국어로 답변하라. "
                    "참고 문서에 없는 사실이나 수치는 추측하지 마라. "
                    "질문에 답할 근거가 부족하면 "
                    "'제공된 문서만으로는 확인할 수 없습니다.'라고 답하라. "
                    "검색 문서에 관련 단어가 있다는 이유만으로 "
                    "답을 만들어내지 마라."
                )
            },
            {
                "role": "user",
                "content": (
                    f"참고 문서:\n{context_text}\n\n"
                    f"질문: {question}"
                )
            }
        ]
    )

    answer = response.choices[0].message.content or ""

    print("\nGPT 답변:")
    print(answer)

    # 4. 실험 결과 기록
    experiment_results.append({
        "experiment": number,
        "question": question,
        "retrieved_documents": retrieved_docs,
        "answer": answer
    })

# 5. JSON 파일 저장
output = {
    "created_at": datetime.now().astimezone().isoformat(),
    "embedding_model": EMBEDDING_MODEL,
    "answer_model": ANSWER_MODEL,
    "vector_db": "Chroma",
    "top_k": TOP_K,
    "experiments": experiment_results
}

Path("results.json").write_text(
    json.dumps(output, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\n==============================")
print("3개 실험 완료!")
print("결과 저장: results.json")
print("==============================")
