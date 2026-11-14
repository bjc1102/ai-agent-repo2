"""새 main.py의 단계 분리·종료 상태와 프롬프트 이전을 검증."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from main import main
from config import BASE_DIR
from prompts.generation import GENERATION_PROMPT, GENERATION_PROMPT_VERSION
from prompts.judge import JUDGE_PROMPT, JUDGE_PROMPT_VERSION


class EntryPointTests(unittest.TestCase):
    """실제 API 없이 사용자가 고른 명령만 실행되는지 확인."""

    def test_generation_and_evaluation_remain_separate_commands(self):
        """평가 명령이 답변을 재생성하지 않고 지정한 결과 경로를 전달."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rag_dir, eval_dir = root / "rag_run", root / "eval_run"
            rag_dir.mkdir()
            eval_dir.mkdir()
            for number in range(1, 11):
                (rag_dir / f"Q{number:02d}.json").write_text(json.dumps({"execution_status": "completed"}))
                (eval_dir / f"Q{number:02d}.json").write_text(json.dumps({
                    "judge": {"evaluation_status": "completed"},
                    "ragas": {name: {"evaluation_status": "completed"} for name in (
                        "context_precision", "context_recall", "faithfulness", "answer_relevancy")},
                }))
            with (patch("rag.pipeline.run_generation", return_value=rag_dir) as generate,
                  patch("evaluation.runner.run_evaluation", new_callable=AsyncMock, return_value=eval_dir) as evaluate,
                  patch("common.api.load_credentials", side_effect=AssertionError("테스트에서 키를 읽을 수 없습니다.")),
                  contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(main(["generate", "--output-dir", str(root)]), 0)
                generate.assert_called_once_with(root)
                evaluate.assert_not_called()
                self.assertEqual(main(["evaluate", str(rag_dir), "--output-dir", str(root)]), 0)
                evaluate.assert_awaited_once_with(rag_dir, root)
                generate.assert_called_once()

    def test_failed_generation_returns_nonzero_and_preserves_records(self):
        """생성 오류가 있는 실행은 실패 종료 코드로 알리고 기록은 보존."""
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            for number in range(1, 11):
                (run_dir / f"Q{number:02d}.json").write_text(json.dumps({
                    "execution_status": "error" if number == 1 else "completed",
                }))
            before = {path.name: path.read_bytes() for path in run_dir.iterdir()}
            with patch("rag.pipeline.run_generation", return_value=run_dir), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["generate"]), 1)
            self.assertEqual(before, {path.name: path.read_bytes() for path in run_dir.iterdir()})

    def test_python_prompt_constants_preserve_the_agreed_text(self):
        """Python 코드의 해시가 아닌 실제 전달 문자열과 기준 문서를 비교."""
        baseline = json.loads((BASE_DIR / "data/rag/baseline.json").read_text(encoding="utf-8"))
        prompt = baseline["generation_prompt"]
        self.assertEqual(prompt["hash_scope"], "prompt_text")
        self.assertEqual(hashlib.sha256(GENERATION_PROMPT.encode()).hexdigest(), prompt["sha256"])
        self.assertEqual(GENERATION_PROMPT_VERSION, "generation_v1")
        self.assertEqual(JUDGE_PROMPT_VERSION, "judge_v1")
        specification = (BASE_DIR / "docs/00-plan/EVALUATION_SPEC.md").read_text(encoding="utf-8")
        agreed = specification.split("### 6.2", 1)[1].split("```text\n", 1)[1].split("```", 1)[0].strip()
        self.assertEqual(JUDGE_PROMPT, agreed)
