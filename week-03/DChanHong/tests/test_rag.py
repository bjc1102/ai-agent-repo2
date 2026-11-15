"""외부 API 없이 검색 순서·실제 입력·오류 기록을 검증한다."""

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from config import EMBEDDING_DIMENSIONS
from rag.pipeline import generation_input, pending_record, process_question, retrieve, run_generation


class FakeCollection:
    """정정 전후의 거리 순서가 다른 검색 결과를 제공한다."""

    def __init__(self):
        """호출된 필터와 개수를 보관한다."""
        self.queries = []

    def query(self, **kwargs):
        """각 정정 구분에서 3개 청크를 반환한다."""
        self.queries.append(kwargs)
        revision = kwargs["where"]["revision"]
        return {
            "ids": [[f"{revision}-{index}" for index in range(3)]],
            "documents": [[f"{revision} 근거 {index}" for index in range(3)]],
            "metadatas": [[{"revision": revision, "pdf_page": index + 1} for index in range(3)]],
            "distances": [[0.9, 0.95, 0.99] if revision == "before" else [0.01, 0.02, 0.03]],
        }


class FakeRecorder:
    """모의 실행에서 논리 호출의 범위만 제공한다."""

    def __init__(self):
        """원본 호출 기록 없이 시작한다."""
        self.calls = []

    @contextmanager
    def scope(self, purpose, question_id):
        """실제 HTTP 연결 없이 호출 블록을 실행한다."""
        yield


class FakeResponse:
    """빈 응답과 정상 응답을 구분할 Responses 모의 객체다."""

    def __init__(self, text):
        """실제 수신한 문자열과 API 상태를 보관한다."""
        self.output_text = text
        self.status = "completed"

    def model_dump(self, **kwargs):
        """원본 응답 저장에 필요한 직렬화 값을 반환한다."""
        return {"status": self.status, "output_text": self.output_text}


class FakeClient:
    """호출 인자를 기록하고 선택한 단계에 모의 오류를 낸다."""

    def __init__(self, text="실제 답변 [E1]", fail_at=None):
        """모의 질문 임베딩과 응답 생성 메서드를 구성한다."""
        self.text = text
        self.fail_at = fail_at
        self.requests = []
        self.embeddings = SimpleNamespace(create=self.embed)
        self.responses = SimpleNamespace(create=self.generate)

    def embed(self, **kwargs):
        """정상 크기의 벡터를 반환하거나 임베딩 오류를 발생시킨다."""
        self.requests.append(("embedding", kwargs))
        if self.fail_at == "embedding":
            raise RuntimeError("mock embedding failure")
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0] * EMBEDDING_DIMENSIONS)])

    def generate(self, **kwargs):
        """실제 생성 요청 인자를 보존한 뒤 모의 응답을 반환한다."""
        self.requests.append(("generation", kwargs))
        if self.fail_at == "generation":
            raise TimeoutError("mock timeout")
        return FakeResponse(self.text)


class RagTests(unittest.TestCase):
    """실행 전 단계에서 보장해야 하는 기록 계약을 확인한다."""

    def execute_case(self, client):
        """임시 폴더에서 한 문항의 모의 실행과 저장을 완료한다."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Q01.json"
            result = process_question(
                pending_record("rag_test", "Q01", "평가 질문"),
                FakeCollection(), client, FakeRecorder(), "고정 생성 지침", path,
            )
            saved = json.loads(path.read_text(encoding="utf-8"))
        return result, saved

    def test_preserves_revision_order_and_question_only_input(self):
        """거리와 무관하게 전후 3개 순서 및 최소 생성 입력을 유지한다."""
        collection = FakeCollection()
        evidence = retrieve(collection, [0.0] * EMBEDDING_DIMENSIONS)
        self.assertEqual([item["label"] for item in evidence], [f"E{i}" for i in range(1, 7)])
        self.assertEqual([item["metadata"]["revision"] for item in evidence], ["before"] * 3 + ["after"] * 3)
        self.assertEqual([item["returned_order"] for item in evidence], list(range(1, 7)))
        self.assertEqual([query["n_results"] for query in collection.queries], [3, 3])
        request = json.loads(generation_input("평가 질문", evidence))
        self.assertEqual(set(request), {"question", "evidence"})
        self.assertTrue(all(set(item) == {"label", "metadata", "text"} for item in request["evidence"]))

    def test_records_the_exact_sent_input(self):
        """저장한 지침·입력·답변이 실제 모의 호출과 동일함을 확인한다."""
        client = FakeClient()
        result, saved = self.execute_case(client)
        generation_request = client.requests[1][1]
        self.assertEqual(result["execution_status"], "completed")
        self.assertEqual(saved["generation"]["input"], generation_request["input"])
        self.assertEqual(saved["generation"]["instructions"], generation_request["instructions"])
        self.assertEqual(saved["generation"]["contexts"], saved["retrieval"]["results"])
        self.assertEqual(saved["generation"]["response"], client.text)
        self.assertEqual(generation_request["reasoning"], {"effort": "none"})
        self.assertEqual(generation_request["max_output_tokens"], 2000)
        self.assertFalse(generation_request["store"])

    def test_empty_response_is_preserved_as_needs_check(self):
        """실제로 받은 빈 문자열을 null이나 정상 답변으로 바꾸지 않는다."""
        result, saved = self.execute_case(FakeClient(text=""))
        self.assertEqual(result["execution_status"], "needs_check")
        self.assertEqual(saved["generation"]["response"], "")
        self.assertEqual(saved["generation"]["execution_status"], "needs_check")

    def test_embedding_failure_skips_generation(self):
        """임베딩 실패를 생성 미실행과 구분해 기록한다."""
        client = FakeClient(fail_at="embedding")
        result, saved = self.execute_case(client)
        self.assertEqual(result["execution_status"], "error")
        self.assertEqual(saved["retrieval"]["execution_status"], "error")
        self.assertEqual(saved["generation"]["execution_status"], "skipped")
        self.assertEqual(saved["generation"]["elapsed_ms"], 0)
        self.assertEqual(len(client.requests), 1)

    def test_generation_error_preserves_retrieval_and_input(self):
        """생성 오류가 나도 당시 근거와 전달한 입력을 잃지 않는다."""
        result, saved = self.execute_case(FakeClient(fail_at="generation"))
        self.assertEqual(result["execution_status"], "error")
        self.assertEqual(saved["retrieval"]["execution_status"], "completed")
        self.assertEqual(len(saved["generation"]["contexts"]), 6)
        self.assertIsNotNone(saved["generation"]["input"])
        self.assertIsNone(saved["generation"]["response"])
        self.assertEqual(saved["generation"]["error"]["code"], "TimeoutError")

    def test_missing_credentials_does_not_create_a_run(self):
        """키 확인 실패를 모의해 API나 실행 결과 폴더가 생기지 않음을 확인한다."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            with patch("rag.pipeline.validate_index", return_value=FakeCollection()), patch(
                "rag.pipeline.make_client", side_effect=ValueError("missing credentials")
            ):
                with self.assertRaisesRegex(ValueError, "missing credentials"):
                    run_generation(target)
            self.assertEqual(list(target.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
