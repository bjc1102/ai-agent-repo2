"""
Week-02 RAG 기본 구현
- Vector Store: ChromaDB (persistent)
- 임베딩: paraphrase-multilingual-mpnet-base-v2 (로컬)
- LLM: Claude claude-sonnet-4-6 (Anthropic API)
- PDF 파싱: pdfplumber
"""

import os
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# ── 설정값 ────────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
PDF_PATH = os.path.join(os.path.dirname(__file__), "data", "2024 알기 쉬운 의료급여제도.pdf")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "medical_aid"


# ── 1. 문서 로드 ───────────────────────────────────────────────
def load_pdf(path: str) -> list[dict]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page": i, "text": text.strip()})
    print(f"[로드] {len(pages)}개 페이지 추출 완료")
    return pages


# ── 2. 청킹 ───────────────────────────────────────────────────
def chunk_pages(pages: list[dict]) -> list[dict]:
    chunks = []
    chunk_id = 0
    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["page"]
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end]
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": chunk_text,
                "metadata": {
                    "source": os.path.basename(PDF_PATH),
                    "page": page_num,
                    "chunk_id": chunk_id,
                },
            })
            chunk_id += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP
    print(f"[청킹] {len(chunks)}개 청크 생성 완료 (크기={CHUNK_SIZE}, 오버랩={CHUNK_OVERLAP})")
    return chunks


# ── 3. 임베딩 & 인덱싱 ────────────────────────────────────────
def build_index(chunks: list[dict], model: SentenceTransformer) -> chromadb.Collection:
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print("[임베딩] 벡터 인코딩 중...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False).tolist()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # 이미 존재하면 삭제 후 재생성 (rag_basic.py 재실행 시 중복 방지)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )
    print(f"[인덱싱] ChromaDB 저장 완료 — 총 {collection.count()}개 청크")
    return collection


# ── 4. 검색 ───────────────────────────────────────────────────
def retrieve(query: str, collection: chromadb.Collection, model: SentenceTransformer, top_k: int = TOP_K) -> list[dict]:
    """쿼리와 가장 유사한 top_k 청크를 반환한다."""
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "metadata": meta,
            "score": round(1 - dist, 4),  # cosine similarity
        })
    return hits


# ── 5. 답변 생성 ───────────────────────────────────────────────
def generate_answer(query: str, hits: list[dict], client: Anthropic) -> str:
    context_blocks = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        context_blocks.append(
            f"[문서 {i}] (출처: {meta['source']}, {meta['page']}페이지, 유사도: {hit['score']})\n{hit['text']}"
        )
    context = "\n\n".join(context_blocks)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=(
            "당신은 의료급여 제도 전문 안내원입니다. "
            "반드시 아래 제공된 문서 내용만을 근거로 답변하세요. "
            "문서에 관련 내용이 없으면 '제공된 문서에서 해당 내용을 찾을 수 없습니다.'라고 답하세요."
        ),
        messages=[
            {
                "role": "user",
                "content": f"{context}\n\n질문: {query}",
            }
        ],
    )
    return response.content[0].text


# ── 메인 ──────────────────────────────────────────────────────
def main():
    # 모델 초기화
    print(f"[모델] SentenceTransformer 로딩: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # 파이프라인
    pages = load_pdf(PDF_PATH)
    chunks = chunk_pages(pages)
    collection = build_index(chunks, embed_model)

    # 테스트 질문
    test_questions = [
        "의료급여 1종과 2종의 차이점은 무엇인가요?",
        "선택병의원 제도란 무엇이고 어떻게 이용하나요?",
        "정신과 의료급여 대상자는 어떤 절차로 입원하나요?",
        "의료급여 수급자가 성형수술을 받으려면 얼마를 내야 하나요?",
    ]

    print("\n" + "=" * 70)
    for i, question in enumerate(test_questions, start=1):
        print(f"\n[질문 {i}] {question}")
        print("-" * 70)

        hits = retrieve(question, collection, embed_model)
        print("▶ 검색된 청크 미리보기:")
        for j, hit in enumerate(hits, start=1):
            meta = hit["metadata"]
            preview = hit["text"][:80].replace("\n", " ")
            print(f"  {j}. p.{meta['page']} (유사도 {hit['score']}) — {preview}…")

        print("\n▶ Claude 답변:")
        answer = generate_answer(question, hits, anthropic_client)
        print(answer)
        print("=" * 70)


if __name__ == "__main__":
    main()
