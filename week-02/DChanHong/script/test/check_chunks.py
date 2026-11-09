import json
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
input_path = BASE_DIR / "output" / "sections.json"
output_path = BASE_DIR / "output" / "chunks.json"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

pdf_files = sorted((BASE_DIR / "docs").glob("*.pdf"))
if len(pdf_files) != 1:
    raise SystemExit("docs 폴더에 PDF가 정확히 1개 있어야 합니다.")

source_name = pdf_files[0].name
sections = json.loads(input_path.read_text(encoding="utf-8"))


def split_text(text):
    """줄을 유지하며 청크를 만들고, 끝부분의 일부 줄을 겹칩니다."""
    lines = text.splitlines(keepends=True)
    start = 0

    while start < len(lines):
        end = start
        size = 0

        # 최대 크기 안에서 줄을 하나씩 추가합니다.
        # 한 줄 자체가 1,200자를 넘으면 그 줄은 통째로 유지합니다.
        while end < len(lines):
            line_size = len(lines[end])
            if end > start and size + line_size > CHUNK_SIZE:
                break
            size += line_size
            end += 1

        chunk_text = "".join(lines[start:end]).strip()
        if chunk_text:
            yield chunk_text

        if end == len(lines):
            break

        # 다음 청크에 재사용할 끝부분의 줄을 선택합니다.
        next_start = end
        overlap_size = 0

        while next_start > start + 1:
            candidate_size = len(lines[next_start - 1])

            if overlap_size + candidate_size > CHUNK_OVERLAP:
                break

            # 겹치는 부분 뒤에 새로운 줄도 들어갈 공간을 확보합니다.
            if (
                overlap_size + candidate_size + len(lines[end])
                > CHUNK_SIZE
            ):
                break

            overlap_size += candidate_size
            next_start -= 1

        start = next_start


chunks = []

for section_index, section in enumerate(sections, start=1):
    for chunk_index, text in enumerate(
        split_text(section["text"]), start=1
    ):
        chunks.append({
            "id": (
                f"dongwon-20261030"
                f"-s{section_index:03d}-c{chunk_index:03d}"
            ),
            "text": text,
            "metadata": {
                "source": source_name,
                "pdf_page": section["pdf_page"],
                "amendment_no": section["amendment_no"],
                "revision": section["revision"],
                "section_index": section_index,
                "chunk_index": chunk_index,
            },
        })

if not chunks:
    raise SystemExit("생성된 청크가 없습니다. 입력 파일을 확인해주세요.")

output_path.write_text(
    json.dumps(chunks, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

counts = Counter(
    chunk["metadata"]["revision"] for chunk in chunks
)
lengths = [len(chunk["text"]) for chunk in chunks]

print(f"입력 구간 수: {len(sections)}")
print(f"전체 청크 수: {len(chunks)}")
print(f"정정 전 청크 수: {counts['before']}")
print(f"정정 후 청크 수: {counts['after']}")
print(f"요약·안내 청크 수: {counts['overview']}")
print(f"최대 청크 길이: {max(lengths)}자")
print(f"1,200자 초과 청크 수: {sum(n > CHUNK_SIZE for n in lengths)}")

# 같은 정정 항목의 전후 청크를 하나씩 확인합니다.
for revision in ("before", "after"):
    sample = next(
        (
            chunk for chunk in chunks
            if chunk["metadata"]["amendment_no"] == 1
            and chunk["metadata"]["revision"] == revision
        ),
        None,
    )

    if sample:
        print(f"\n===== 주1 / {revision} 첫 청크 =====")
        print(f"ID: {sample['id']}")
        print(f"PDF 페이지: {sample['metadata']['pdf_page']}")
        print(sample["text"])

print(f"\n저장 위치: {output_path}")