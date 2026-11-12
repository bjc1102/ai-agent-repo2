
from pathlib import Path
import json

# 1. 설정
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# 2. GuardDuty 문서 읽기
input_path = Path("data/guardduty.txt")
text = input_path.read_text(encoding="utf-8")

print(f"원본 문서 길이: {len(text):,}자")

# 3. 청킹 함수
def split_text(text, chunk_size, overlap):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks

# 4. 청킹 실행
chunks = split_text(
    text,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

# 5. 문서 출처와 청크 번호 기록
results = []

for i, chunk in enumerate(chunks):
    results.append({
        "id": f"chunk_{i:03d}",
        "source": "AWS GuardDuty EC2 Finding Types",
        "url": (
            "https://docs.aws.amazon.com/guardduty/"
            "latest/ug/guardduty_finding-types-ec2.html"
        ),
        "chunk_index": i,
        "text": chunk
    })

# 6. JSON 파일로 저장
output_path = Path("data/chunks.json")

output_path.write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

# 7. 결과 출력
print(f"생성된 청크 수: {len(chunks)}개")
print(f"저장 위치: {output_path}")

print("\n===== 첫 번째 청크 =====")
print(chunks[0])

print("\n===== 두 번째 청크 =====")
print(chunks[1] if len(chunks) > 1 else "없음")
