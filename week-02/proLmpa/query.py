"""
query.py — rag_basic.py 실행 후 저장된 ChromaDB 컬렉션을 불러와
사용자 질문에 대화형으로 답변하는 스크립트.

사전 조건:
  python rag_basic.py  # chroma_db/ 디렉토리가 생성돼 있어야 함

사용법:
  python query.py
  python query.py --top-k 5   # 검색 청크 수 조정
"""

import os
import argparse
import chromadb
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv

# rag_basic.py 의 공유 설정값 재사용
from rag_basic import (
    EMBEDDING_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
    TOP_K,
    retrieve,
    generate_answer,
)

load_dotenv()


def load_collection() -> chromadb.Collection:
    """디스크에 저장된 ChromaDB 컬렉션을 읽기 전용으로 불러온다."""
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(
            f"'{CHROMA_PATH}' 디렉토리가 없습니다. 먼저 rag_basic.py 를 실행하세요."
        )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"[컬렉션 로드] '{COLLECTION_NAME}' — {collection.count()}개 청크")
    return collection


def main():
    parser = argparse.ArgumentParser(description="의료급여 RAG 질의 도구")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="검색 청크 수 (기본: %(default)s)")
    args = parser.parse_args()

    print(f"[모델] SentenceTransformer 로딩: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    collection = load_collection()

    print("\n의료급여 RAG 질의 시스템 준비 완료.")
    print("질문을 입력하세요. 종료하려면 'q' 또는 빈 줄을 입력하세요.\n")

    while True:
        try:
            query = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not query or query.lower() in {"q", "quit", "exit"}:
            print("종료합니다.")
            break

        hits = retrieve(query, collection, embed_model, top_k=args.top_k)

        print("\n▶ 검색된 청크:")
        for i, hit in enumerate(hits, start=1):
            meta = hit["metadata"]
            preview = hit["text"][:80].replace("\n", " ")
            print(f"  {i}. p.{meta['page']} (유사도 {hit['score']}) — {preview}…")

        print("\n▶ Claude 답변:")
        answer = generate_answer(query, hits, anthropic_client)
        print(answer)
        print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
