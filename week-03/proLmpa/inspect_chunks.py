"""
inspect_chunks.py — Week-03 one-shot helper
ChromaDB에 저장된 모든 청크를 ID, 페이지, 텍스트 미리보기와 함께 출력한다.
eval_questions.json 의 ground_truth_chunk_ids 를 채우기 위해 한 번만 실행하면 된다.

실행:
    python inspect_chunks.py [--page <page_num>] [--all]

기본값: 주요 평가 대상 페이지(13, 17, 18, 19)의 청크만 출력.
--all : 전체 청크 출력 (수백 개일 수 있음)
--page N : 특정 페이지만 출력
"""

import sys
import os
import argparse

# week-02 rag_basic 모듈 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "week-02", "proLmpa"))

import chromadb
from sentence_transformers import SentenceTransformer
from rag_basic import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)


def inspect(pages_filter: list[int] | None = None, show_all: bool = False) -> None:
    collection = get_collection()
    total = collection.count()
    print(f"[ChromaDB] 컬렉션: {COLLECTION_NAME}  총 청크 수: {total}\n")

    # 전체 데이터 가져오기 (ID 순 정렬)
    result = collection.get(include=["documents", "metadatas"])
    ids = result["ids"]
    docs = result["documents"]
    metas = result["metadatas"]

    # 청크 ID 숫자 기준으로 정렬
    items = sorted(zip(ids, docs, metas), key=lambda x: int(x[0].split("_")[1]))

    target_pages = set(pages_filter) if pages_filter else {13, 17, 18, 19}

    printed = 0
    for chunk_id, text, meta in items:
        page = meta.get("page", -1)
        if not show_all and page not in target_pages:
            continue

        preview = text[:120].replace("\n", " ")
        print(f"ID: {chunk_id:<12}  page: {page:>3}  len: {len(text):>4}자")
        print(f"  미리보기: {preview}…")
        print()
        printed += 1

    if printed == 0:
        print("(해당 페이지의 청크가 없습니다)")
    else:
        print(f"--- {printed}개 청크 출력됨 (전체 {total}개 중) ---")


def main() -> None:
    parser = argparse.ArgumentParser(description="ChromaDB 청크 검사 도구")
    parser.add_argument("--page", type=int, help="특정 페이지 번호만 출력")
    parser.add_argument("--all", action="store_true", help="전체 청크 출력")
    args = parser.parse_args()

    if args.all:
        inspect(show_all=True)
    elif args.page:
        inspect(pages_filter=[args.page])
    else:
        # 기본: 평가 질문 관련 페이지만 출력
        print("=== 평가 질문 관련 페이지 청크 (p.13, 17, 18, 19) ===\n")
        inspect()


if __name__ == "__main__":
    main()
