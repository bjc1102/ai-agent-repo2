import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
input_path = BASE_DIR / "output" / "pdf_text.txt"
output_path = BASE_DIR / "output" / "sections.json"

# 페이지 표시와 정정 제목을 찾는 규칙입니다.
page_pattern = re.compile(r"===== PDF 페이지 (\d+) =====")
revision_pattern = re.compile(r"\(주\s*(\d+)\)\s*정정\s*(전|후)")

sections = []
buffer = []

pdf_page = 0
amendment_no = 0
revision = "overview"


def save_section():
    """지금까지 모은 본문에 페이지·정정 정보를 붙여 저장합니다."""
    text = "\n".join(buffer).strip()

    if text:
        sections.append({
            "pdf_page": pdf_page,
            "amendment_no": amendment_no,
            "revision": revision,
            "text": text,
        })

    buffer.clear()


for raw_line in input_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()

    # 새 페이지를 만나면 이전 페이지의 본문을 저장합니다.
    page_match = page_pattern.fullmatch(line)
    if page_match:
        save_section()
        pdf_page = int(page_match.group(1))
        # 정정 항목과 전후 구분은 다음 페이지에도 이어집니다.
        continue

    # 이 문서의 첫 페이지인 목차를 제외합니다.
    if pdf_page < 2:
        continue

    # 반복되는 바닥글을 제외합니다.
    if line.startswith("전자공시시스템 dart.fss.or.kr"):
        continue

    # 새 정정 제목을 만나면 이전 구간을 저장하고 구분을 바꿉니다.
    revision_match = revision_pattern.fullmatch(line)
    if revision_match:
        save_section()
        amendment_no = int(revision_match.group(1))
        revision = (
            "before" if revision_match.group(2) == "전" else "after"
        )
        continue

    # 마지막 공통 안내문은 정정 본문과 별도로 구분합니다.
    if line.startswith("위 정정사항외"):
        save_section()
        amendment_no = 0
        revision = "overview"

    # 표의 가로 간격은 유지하고, 연속된 빈 줄만 줄입니다.
    if line:
        buffer.append(raw_line.rstrip())
    elif buffer and buffer[-1] != "":
        buffer.append("")

# 마지막 구간도 저장합니다.
save_section()

before = {
    item["amendment_no"]
    for item in sections
    if item["revision"] == "before"
}
after = {
    item["amendment_no"]
    for item in sections
    if item["revision"] == "after"
}

output_path.write_text(
    json.dumps(sections, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"정정 전 항목 수: {len(before)}")
print(f"정정 후 항목 수: {len(after)}")
print(f"정정 전만 있는 항목: {sorted(before - after)}")
print(f"정정 후만 있는 항목: {sorted(after - before)}")

print("\nPDF 5페이지 구간:")
for item in sections:
    if item["pdf_page"] == 5:
        print(f"- 주{item['amendment_no']}: {item['revision']}")

page_91_count = sum(item["pdf_page"] == 91 for item in sections)
print(f"\nPDF 91페이지 본문 구간 수: {page_91_count}")
print(f"전체 페이지별 구간 수: {len(sections)}")
print(f"저장 위치: {output_path}")