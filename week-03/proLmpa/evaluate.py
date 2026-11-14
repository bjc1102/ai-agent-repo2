"""
evaluate.py — Week-03 RAG 평가 파이프라인

단계:
  1. week-02 rag_basic 모듈 임포트 (retrieve, generate_answer 등)
  2. eval_questions.json 로드
  3. 각 질문에 대해 retrieve(top_k=5) + generate_answer 실행
  4. ID-based Recall@5 계산 (no LLM)
  5. 답변 가능성(answerability) binary 판정
  6. Ragas 지표 계산 (Faithfulness, FactualCorrectness, ContextRecall, ResponseRelevancy)
  7. results/ 디렉토리에 raw_results.json + eval_results.json 저장

실행:
    python evaluate.py

사전 조건:
  - inspect_chunks.py 실행 후 eval_questions.json 의 ground_truth_chunk_ids 를 채워넣을 것
  - ANTHROPIC_API_KEY 환경 변수 설정 (.env 파일 또는 쉘 환경)
"""

import sys
import os
import json
import math
from pathlib import Path

# ── week-02 모듈 경로 추가 ────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "week-02", "proLmpa"))

from dotenv import load_dotenv
load_dotenv()

from rag_basic import (
    retrieve,
    generate_answer,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

import chromadb
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

# ── 설정 ─────────────────────────────────────────────────────
EVAL_TOP_K = 5  # week-02 는 TOP_K=3, 평가는 5로 확장
RESULTS_DIR = Path(__file__).parent / "results"
QUESTIONS_FILE = Path(__file__).parent / "eval_questions.json"

# 답변 거절 판정 키워드
REFUSAL_PHRASES = [
    "확인할 수 없",
    "찾을 수 없",
    "해당 내용이 없",
    "제공된 문서에서",
]

# ── 프롬프트 비교 (Gap 2: 성형수술 추론 억제) ─────────────────
# week-02 원본 프롬프트는 rag_basic.generate_answer() 내부에 있음.
# Strict 버전은 추론·추정·외부 문의처를 명시적으로 금지.
STRICT_SYSTEM_PROMPT = (
    "당신은 의료급여 제도 전문 안내원입니다. "
    "반드시 아래 제공된 문서 내용만을 근거로 답변하세요. "
    "문서에 관련 내용이 없으면 '제공된 문서에서 해당 내용을 찾을 수 없습니다.'라고만 답하고, "
    "추론·추정·가능성 제시 및 외부 문의처 안내 등 문서 외 내용을 절대 덧붙이지 마세요."
)


# ── 초기화 ────────────────────────────────────────────────────
def init_components():
    """SentenceTransformer, ChromaDB 컬렉션, Anthropic 클라이언트 초기화"""
    print(f"[초기화] SentenceTransformer 로딩: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"[초기화] ChromaDB 연결: {CHROMA_PATH} / {COLLECTION_NAME}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"[초기화] 총 청크 수: {collection.count()}")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY 환경 변수가 설정되지 않았습니다.")
    anthropic_client = Anthropic(api_key=api_key)

    return embed_model, collection, anthropic_client


# ── 메트릭 계산 (LLM 불필요) ──────────────────────────────────
def calc_id_recall(retrieved_ids: list[str], ground_truth_ids: list[str]) -> float | None:
    """ID-based Recall@K: 정답 청크 중 검색된 비율"""
    if not ground_truth_ids:
        return None  # Q4: 정답 청크 없음
    retrieved_set = set(retrieved_ids)
    gt_set = set(ground_truth_ids)
    return len(retrieved_set & gt_set) / len(gt_set)


def calc_answerability(answer: str, expected_answerable) -> dict:
    """
    답변 거절 여부를 키워드로 판정.
    Returns:
        {
          "refused": bool,          # 거절 문구 포함 여부
          "correct": bool | None,   # 예상과 일치 여부 (partial 은 None)
        }
    """
    refused = any(phrase in answer for phrase in REFUSAL_PHRASES)
    if expected_answerable is True:
        correct = not refused  # 답변해야 하는 질문인데 거절하면 오류
    elif expected_answerable is False:
        correct = refused  # 거절해야 하는 질문인데 답변하면 오류
    else:
        correct = None  # partial: 인간 판단 필요
    return {"refused": refused, "correct": correct}


def build_context_string(hits: list[dict]) -> str:
    """generate_answer 내부와 동일한 컨텍스트 문자열 생성 (기록용)"""
    context_blocks = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        context_blocks.append(
            f"[문서 {i}] (출처: {meta['source']}, {meta['page']}페이지, 유사도: {hit['score']})\n{hit['text']}"
        )
    return "\n\n".join(context_blocks)


def generate_answer_strict(query: str, hits: list[dict], client: Anthropic) -> str:
    """STRICT_SYSTEM_PROMPT 로 문서 외 추론·추정을 금지한 답변 생성."""
    context = build_context_string(hits)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=STRICT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"{context}\n\n질문: {query}"}],
    )
    return response.content[0].text


# ── Ragas 평가 ────────────────────────────────────────────────
def run_ragas_evaluation(samples_data: list[dict], llm, embeddings) -> dict[str, dict]:
    """
    Ragas 지표를 계산한다.
    samples_data: list of {id, user_input, retrieved_contexts, response, reference, include_context_recall}

    Returns: {question_id: {metric_name: score_or_nan, ...}}
    """
    from ragas import evaluate
    from ragas import EvaluationDataset
    from ragas.metrics import Faithfulness, FactualCorrectness, ResponseRelevancy, ContextRecall  # noqa: deprecated import path, LangchainLLMWrapper incompatible with ragas.metrics.collections
    from ragas.dataset_schema import SingleTurnSample

    metric_results: dict[str, dict] = {}

    # 각 질문 ID를 초기화
    for s in samples_data:
        metric_results[s["id"]] = {
            "faithfulness": None,
            "faithfulness_na_reason": None,
            "factual_correctness": None,
            "response_relevancy": None,
            "context_recall": None,
            "context_recall_na_reason": None,
        }

    # --- Faithfulness, FactualCorrectness, ResponseRelevancy ---
    # Q4 (unanswerable): Faithfulness → N/A (거절 답변에 주장 없음)
    # 나머지는 모두 포함
    full_samples = []
    full_ids = []
    for s in samples_data:
        sample = SingleTurnSample(
            user_input=s["user_input"],
            retrieved_contexts=s["retrieved_contexts"],
            response=s["response"],
            reference=s["reference"],
        )
        full_samples.append(sample)
        full_ids.append(s["id"])

    if full_samples:
        print("[Ragas] FactualCorrectness, AnswerRelevancy 계산 중...")
        dataset_full = EvaluationDataset(samples=full_samples)
        result_full = evaluate(
            dataset_full,
            metrics=[FactualCorrectness(), ResponseRelevancy()],
            llm=llm,
            embeddings=embeddings,
        )
        df_full = result_full.to_pandas()
        for i, qid in enumerate(full_ids):
            if i < len(df_full):
                row = df_full.iloc[i]
                # Ragas 0.4.4 actual column names (confirmed via DataFrame inspection):
                # FactualCorrectness → "factual_correctness(mode=f1)"
                # ResponseRelevancy → "answer_relevancy"
                fc_val = row.get("factual_correctness(mode=f1)")
                metric_results[qid]["factual_correctness"] = (
                    None if (isinstance(fc_val, float) and math.isnan(fc_val)) else fc_val
                )
                ar_val = row.get("answer_relevancy")
                metric_results[qid]["response_relevancy"] = (
                    None if (isinstance(ar_val, float) and math.isnan(ar_val)) else ar_val
                )

    # --- Faithfulness: Q4 제외 ---
    faith_samples = []
    faith_ids = []
    for s in samples_data:
        if not s.get("answerable", True) is False:  # Q4(answerable=False) 제외
            faith_samples.append(SingleTurnSample(
                user_input=s["user_input"],
                retrieved_contexts=s["retrieved_contexts"],
                response=s["response"],
                reference=s["reference"],
            ))
            faith_ids.append(s["id"])
        else:
            metric_results[s["id"]]["faithfulness_na_reason"] = (
                "거절 답변(unanswerable question)에는 검증할 주장이 없으므로 N/A"
            )

    if faith_samples:
        print("[Ragas] Faithfulness 계산 중...")
        dataset_faith = EvaluationDataset(samples=faith_samples)
        result_faith = evaluate(dataset_faith, metrics=[Faithfulness()], llm=llm, embeddings=embeddings)
        df_faith = result_faith.to_pandas()
        for i, qid in enumerate(faith_ids):
            if i < len(df_faith):
                row = df_faith.iloc[i]
                val = row.get("faithfulness")
                metric_results[qid]["faithfulness"] = (
                    None if (isinstance(val, float) and math.isnan(val)) else val
                )

    # --- ContextRecall: Q4 제외 (정답 청크 없음 = reference 도 없음) ---
    cr_samples = []
    cr_ids = []
    for s in samples_data:
        if s.get("answerable", True) is not False and s.get("include_context_recall", True):
            cr_samples.append(SingleTurnSample(
                user_input=s["user_input"],
                retrieved_contexts=s["retrieved_contexts"],
                response=s["response"],
                reference=s["reference"],
            ))
            cr_ids.append(s["id"])
        else:
            metric_results[s["id"]]["context_recall_na_reason"] = (
                "정답 청크(ground truth)가 없는 unanswerable 질문 → N/A"
            )

    if cr_samples:
        print("[Ragas] ContextRecall 계산 중...")
        dataset_cr = EvaluationDataset(samples=cr_samples)
        result_cr = evaluate(dataset_cr, metrics=[ContextRecall()], llm=llm, embeddings=embeddings)
        df_cr = result_cr.to_pandas()
        for i, qid in enumerate(cr_ids):
            if i < len(df_cr):
                row = df_cr.iloc[i]
                val = row.get("context_recall")
                metric_results[qid]["context_recall"] = (
                    None if (isinstance(val, float) and math.isnan(val)) else val
                )

    return metric_results


# ── 메인 파이프라인 ───────────────────────────────────────────
def main():
    # 결과 디렉토리 생성
    RESULTS_DIR.mkdir(exist_ok=True)

    # 컴포넌트 초기화
    embed_model, collection, anthropic_client = init_components()

    # Ragas LLM 설정 (Claude claude-sonnet-4-6)
    # 참고: 생성 모델과 동일한 모델을 판정자로 사용하므로 self-judging bias 위험 있음
    print("[Ragas] LLM 초기화: claude-sonnet-4-6 (langchain-anthropic)")
    from langchain_anthropic import ChatAnthropic
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_core.embeddings import Embeddings as LCEmbeddings

    # bypass_temperature=True: claude-sonnet-4-6 rejects temperature as an extra parameter.
    # Ragas 0.4.4 adds bypass_temperature for exactly this case (same fix as OpenAI o1).
    ragas_llm = LangchainLLMWrapper(
        ChatAnthropic(
            model="claude-sonnet-4-6",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        ),
        bypass_temperature=True,
    )

    # Ragas 임베딩 설정: 이미 로드된 SentenceTransformer를 재사용 (추가 패키지 불필요)
    class _STEmbeddings(LCEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return embed_model.encode(texts, show_progress_bar=False).tolist()
        def embed_query(self, text: str) -> list[float]:
            return embed_model.encode([text], show_progress_bar=False).tolist()[0]

    ragas_embeddings = LangchainEmbeddingsWrapper(_STEmbeddings())

    # 평가 질문 로드
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    print(f"\n[평가 시작] {len(questions)}개 질문, top_k={EVAL_TOP_K}\n" + "=" * 70)

    raw_results = []
    ragas_input = []
    all_hits: dict[str, list[dict]] = {}  # hits 캐시 (Q4 prompt comparison 에 재사용)

    # ── Step 2-4: 각 질문에 대해 retrieve + generate ───────────
    for q in questions:
        qid = q["id"]
        query = q["question"]
        print(f"\n[{qid}] {query[:60]}…")

        # 검색 (top_k=5)
        hits = retrieve(query, collection, embed_model, top_k=EVAL_TOP_K)
        all_hits[qid] = hits

        retrieved_chunk_ids = [h["metadata"].get("chunk_id") for h in hits]
        # chunk_id 메타데이터가 int일 수 있으므로 문자열로 변환
        retrieved_chunk_ids_str = [f"chunk_{cid}" if isinstance(cid, int) else str(cid) for cid in retrieved_chunk_ids]
        retrieved_pages = [h["metadata"].get("page") for h in hits]
        retrieved_scores = [h["score"] for h in hits]

        print(f"  검색된 청크: {retrieved_chunk_ids_str} (페이지: {retrieved_pages})")

        # 컨텍스트 문자열 (generate_answer 에 전달되는 것과 동일)
        context_string = build_context_string(hits)

        # 답변 생성
        answer = generate_answer(query, hits, anthropic_client)
        print(f"  답변 길이: {len(answer)}자")

        # ID-based Recall@5
        gt_ids = q.get("ground_truth_chunk_ids", [])
        id_recall = calc_id_recall(retrieved_chunk_ids_str, gt_ids)
        if id_recall is None:
            print(f"  ID Recall@5: N/A (unanswerable — 정답 청크 없음)")
        else:
            print(f"  ID Recall@5: {id_recall:.3f} (GT: {gt_ids})")

        # Answerability binary
        answerability = calc_answerability(answer, q.get("answerable"))
        print(f"  거절 여부: {answerability['refused']}  예상 일치: {answerability['correct']}")

        # raw 결과 기록
        raw_results.append({
            "id": qid,
            "type": q.get("type"),
            "question": query,
            "retrieved_chunk_ids": retrieved_chunk_ids_str,
            "retrieved_pages": retrieved_pages,
            "retrieved_scores": retrieved_scores,
            "context_string": context_string,
            "context_note": "retrieved_contexts == generation_input (no reranking)",
            "answer": answer,
            "ground_truth_chunk_ids": gt_ids,
            "id_recall_at_5": id_recall,
            "answerability": answerability,
            "config": {
                "top_k": EVAL_TOP_K,
                "embedding_model": EMBEDDING_MODEL,
                "generation_model": "claude-sonnet-4-6",
                "ragas_judge_model": "claude-sonnet-4-6",
            },
        })

        # Ragas 입력 준비
        ragas_input.append({
            "id": qid,
            "user_input": query,
            "retrieved_contexts": [h["text"] for h in hits],
            "response": answer,
            "reference": q.get("reference", ""),
            "answerable": q.get("answerable"),
            "include_context_recall": bool(gt_ids),  # 정답 청크가 있을 때만 ContextRecall
        })

    # raw 결과 저장
    raw_path = RESULTS_DIR / "raw_results.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {raw_path}")

    # ── Step 4a: Q4 프롬프트 비교 (Gap 2 — 성형수술 추론 억제) ──
    # week-02 원본 프롬프트 vs STRICT_SYSTEM_PROMPT 비교.
    # 동일한 retrieved hits 를 재사용하므로 추가 검색 없이 생성만 반복.
    print("\n[프롬프트 비교] Q4 성형수술 — strict prompt 재실행...")
    q4_raw = next(r for r in raw_results if r["id"] == "Q4")
    q4_strict_answer = generate_answer_strict(q4_raw["question"], all_hits["Q4"], anthropic_client)
    refused_strict = any(p in q4_strict_answer for p in REFUSAL_PHRASES)
    print(f"  strict 거절 여부: {refused_strict}")
    print(f"  strict 답변 길이: {len(q4_strict_answer)}자")

    q4_comparison = {
        "question": q4_raw["question"],
        "retrieved_chunk_ids": q4_raw["retrieved_chunk_ids"],
        "original_system_prompt": (
            "당신은 의료급여 제도 전문 안내원입니다. "
            "반드시 아래 제공된 문서 내용만을 근거로 답변하세요. "
            "문서에 관련 내용이 없으면 '제공된 문서에서 해당 내용을 찾을 수 없습니다.'라고 답하세요."
        ),
        "strict_system_prompt": STRICT_SYSTEM_PROMPT,
        "answer_original": q4_raw["answer"],
        "answer_strict": q4_strict_answer,
        "out_of_document_in_original": [
            "미용 목적 성형수술은 일반적으로 비급여 항목에 해당할 가능성이 높으나 (외부 의료 지식 추론)",
            "보건복지상담센터 ☎ 129 (검색된 청크에 없는 외부 연락처)",
        ],
        "refused_original": q4_raw["answerability"]["refused"],
        "refused_strict": refused_strict,
    }
    comparison_path = RESULTS_DIR / "q4_prompt_comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(q4_comparison, f, ensure_ascii=False, indent=2)
    print(f"[저장] {comparison_path}")

    # ── Step 5: Ragas 평가 ────────────────────────────────────
    print("\n[Ragas] 지표 계산 시작…")
    ragas_scores = run_ragas_evaluation(ragas_input, ragas_llm, ragas_embeddings)

    # ── Step 6: eval_results 조합 ─────────────────────────────
    eval_results = []
    for raw in raw_results:
        qid = raw["id"]
        scores = ragas_scores.get(qid, {})
        eval_results.append({
            "id": qid,
            "type": raw["type"],
            "question": raw["question"],
            "id_recall_at_5": raw["id_recall_at_5"],
            "answerability_refused": raw["answerability"]["refused"],
            "answerability_correct": raw["answerability"]["correct"],
            "faithfulness": scores.get("faithfulness"),
            "faithfulness_na_reason": scores.get("faithfulness_na_reason"),
            "factual_correctness": scores.get("factual_correctness"),
            "response_relevancy": scores.get("response_relevancy"),
            "context_recall": scores.get("context_recall"),
            "context_recall_na_reason": scores.get("context_recall_na_reason"),
            "human_judgment": None,  # 인간이 직접 채워넣을 필드
        })

    eval_path = RESULTS_DIR / "eval_results.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    print(f"[저장] {eval_path}")

    # ── 요약 출력 ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("평가 결과 요약")
    print("=" * 70)
    header = f"{'ID':<4} {'ID Recall@5':>12} {'Faithfulness':>14} {'FactualCorr':>12} {'RespRel':>9} {'CtxRecall':>10}"
    print(header)
    print("-" * 70)
    for r in eval_results:
        def fmt(v):
            if v is None:
                return "   N/A"
            return f"{v:>6.3f}"
        print(
            f"{r['id']:<4} "
            f"{fmt(r['id_recall_at_5']):>12} "
            f"{fmt(r['faithfulness']):>14} "
            f"{fmt(r['factual_correctness']):>12} "
            f"{fmt(r['response_relevancy']):>9} "
            f"{fmt(r['context_recall']):>10}"
        )
    print("=" * 70)
    print(f"\n결과 파일:\n  {raw_path}\n  {eval_path}")
    print("\n다음 단계: 원본 PDF와 답변을 비교하여 eval_results.json 의 human_judgment 필드를 채워넣으세요.")


if __name__ == "__main__":
    main()
