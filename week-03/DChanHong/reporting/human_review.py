"""사람 검토 양식을 만들고 완료한 검토를 새 이력으로 추가."""

from pathlib import Path

from common.utils import new_id, write_json
from schemas import EvaluationRecord, HumanReview
from .summary import DIMENSIONS, derive, save_evaluation, summarize


def review_template(run_dir: Path, question_id: str, output: Path) -> Path:
    """실제 판정을 채우지 않은 별도 사람 검토 작성 양식을 생성."""
    EvaluationRecord.model_validate_json((Path(run_dir) / f"{question_id}.json").read_text(encoding="utf-8"))
    if Path(output).exists():
        raise FileExistsError("기존 검토 파일을 덮어쓰지 않습니다.")
    template = {"review_id": new_id("review"), "reviewer": "", "reviewed_at": None,
                **{d: {"verdict": None, "reason": "", "basis_refs": []} for d in DIMENSIONS},
                "source_checks": [], "disagreements": [], "notes": ""}
    write_json(output, template)
    return Path(output)


def import_review(run_dir: Path, question_id: str, review_path: Path) -> Path:
    """사람이 채운 검토만 검증해 새 이력으로 추가하고 요약을 갱신."""
    path = Path(run_dir) / f"{question_id}.json"
    record = EvaluationRecord.model_validate_json(path.read_text(encoding="utf-8")).model_dump(mode="json")
    review = HumanReview.model_validate_json(Path(review_path).read_text(encoding="utf-8")).model_dump(mode="json")
    if any(old["review_id"] == review["review_id"] for old in record["human_reviews"]):
        raise ValueError("검토 ID가 이미 존재합니다. 수정 내용은 새 ID로 추가하세요.")
    refs = {s["source_id"] for s in record["golden"]["golden_sources"]}
    refs.update(c["criterion_id"] for c in record["golden"]["golden_criteria"])
    refs.update(c["label"] for c in record["rag"]["generation"]["contexts"])
    for name in DIMENSIONS:
        if not set(review[name]["basis_refs"]) <= refs:
            raise ValueError("검토에서 입력에 없는 근거 식별자를 사용했습니다.")
    document_ids = {s["document_id"] for s in record["golden"]["golden_sources"]}
    if any(s["document_id"] not in document_ids for s in review["source_checks"]):
        raise ValueError("검토 문서 식별자가 골든 문서와 다릅니다.")
    record["human_reviews"].append(review)
    record["derived"] = derive(record, record["derived"]["eval_elapsed_ms"])
    save_evaluation(path, record)
    summarize(Path(run_dir))
    return path
