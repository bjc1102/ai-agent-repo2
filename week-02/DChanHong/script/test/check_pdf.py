from pathlib import Path

from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parents[2]
pdf_files = sorted((BASE_DIR / "docs").glob("*.pdf"))

if len(pdf_files) != 1:
    raise SystemExit(
        f"docs 폴더에 PDF가 정확히 1개 있어야 합니다. 현재: {len(pdf_files)}개"
    )

pdf_path = pdf_files[0]
reader = PdfReader(pdf_path)
page_count = len(reader.pages)

output_dir = BASE_DIR / "output"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "pdf_text.txt"

empty_pages = []
short_pages = []
replacement_pages = []
total_chars = 0

# 앞부분, 중간, 마지막 페이지의 일부를 터미널에 표시합니다.
sample_pages = {1, (page_count + 1) // 2, page_count}

print(f"대상 PDF: {pdf_path.name}")
print(f"전체 페이지 수: {page_count}", flush=True)

with output_path.open("w", encoding="utf-8") as output:
    for page_number, page in enumerate(reader.pages, start=1):
        # 원문의 줄·열 배치를 가능한 한 유지하며 추출합니다.
        text = page.extract_text(extraction_mode="layout") or ""
        char_count = len("".join(text.split()))
        total_chars += char_count

        if char_count == 0:
            empty_pages.append(page_number)
        elif char_count < 50:
            short_pages.append(page_number)

        if "\ufffd" in text:
            replacement_pages.append(page_number)

        output.write(f"\n\n===== PDF 페이지 {page_number} =====\n\n")
        output.write(text)

        if page_number in sample_pages:
            print(f"\n[PDF 페이지 {page_number} 미리보기]")
            print(text.strip()[:800] or "(추출된 텍스트 없음)")

        if page_number % 50 == 0:
            print(f"\n진행: {page_number}/{page_count}페이지", flush=True)

print("\n===== 추출 결과 =====")
print(f"전체 페이지 수: {page_count}")
print(f"전체 글자 수(공백 제외): {total_chars:,}")
print(f"텍스트가 없는 페이지: {empty_pages}")
print(f"텍스트가 50자 미만인 페이지: {short_pages}")
print(f"대체 문자(�)가 포함된 페이지: {replacement_pages}")
print(f"전체 추출 텍스트 저장 위치: {output_path}")
