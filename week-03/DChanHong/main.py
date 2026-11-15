"""프로젝트의 검증·생성·평가·사람 검토 명령을 연결하는 진입점."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from common.validation import validate


def main(argv: list[str] | None = None) -> int:
    """선택한 명령만 실행하고 다음 단계는 사용자가 직접 선택하도록 함."""
    parser = argparse.ArgumentParser(description="동원파츠 RAG 10문항 실행·평가")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="API 호출 없는 준비 상태 검증")
    prepare = commands.add_parser("prepare-index", help="기존 색인 복사·검증 (API 없음)")
    prepare.add_argument("--source", type=Path)
    generate = commands.add_parser("generate", help="10개 질문의 응답 생성 (유료 API)")
    generate.add_argument("--output-dir", type=Path)
    evaluate = commands.add_parser("evaluate", help="저장한 10개 응답 평가 (유료 API)")
    evaluate.add_argument("rag_run_dir", type=Path)
    evaluate.add_argument("--output-dir", type=Path)
    summary = commands.add_parser("summarize", help="상세 JSON으로 요약 재생성")
    summary.add_argument("eval_run_dir", type=Path)
    for command in ("review-template", "import-review"):
        review = commands.add_parser(command, help="사람 검토 양식 생성" if command == "review-template" else "완료된 사람 검토 이력 추가")
        review.add_argument("eval_run_dir", type=Path)
        review.add_argument("question_id", choices=[f"Q{n:02d}" for n in range(1, 11)])
        review.add_argument("review_file", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            validate()
        elif args.command == "prepare-index":
            from rag.index import prepare_index
            print(prepare_index(args.source))
        elif args.command == "generate":
            from rag.pipeline import run_generation
            path = run_generation(args.output_dir)
            print(path)
            records = [json.loads(p.read_text(encoding="utf-8")) for p in path.glob("Q*.json")]
            return 0 if len(records) == 10 and all(r["execution_status"] == "completed" for r in records) else 1
        elif args.command == "evaluate":
            from evaluation.runner import run_evaluation
            path = asyncio.run(run_evaluation(args.rag_run_dir, args.output_dir))
            print(path)
            records = [json.loads(p.read_text(encoding="utf-8")) for p in path.glob("Q*.json")]
            return 0 if all(r["judge"]["evaluation_status"] == "completed" and all(m["evaluation_status"] in ("completed", "not_applicable") for m in r["ragas"].values()) for r in records) else 1
        elif args.command == "summarize":
            from reporting.summary import summarize
            print(summarize(args.eval_run_dir))
        else:
            from reporting.human_review import import_review, review_template
            if args.command == "review-template":
                print(review_template(args.eval_run_dir, args.question_id, args.review_file))
            elif args.command == "import-review":
                print(import_review(args.eval_run_dir, args.question_id, args.review_file))
        return 0
    except (OSError, ValueError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
