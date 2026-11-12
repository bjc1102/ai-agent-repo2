
import requests
from bs4 import BeautifulSoup
from pathlib import Path

# 1. AWS GuardDuty 공식 문서 주소
URL = "https://docs.aws.amazon.com/guardduty/latest/ug/guardduty_finding-types-ec2.html"

# 2. 웹페이지 다운로드
response = requests.get(URL, timeout=30)
response.raise_for_status()

# 3. HTML 분석
soup = BeautifulSoup(response.text, "html.parser")

# 불필요한 태그 제거
for tag in soup(["script", "style", "nav", "footer"]):
    tag.decompose()

# 4. 문서 본문 찾기
main = (
    soup.find("main")
    or soup.find("article")
    or soup.find(id="main-content")
    or soup.find(id="main-col-body")
)

# 본문을 찾지 못하면 body 사용
if main is None:
    print("본문 태그를 찾지 못해 전체 HTML에서 추출합니다.")
    main = soup.body

if main is None:
    raise RuntimeError("HTML 본문이 존재하지 않습니다.")

# 5. 텍스트 추출
text = main.get_text(separator="\n", strip=True)

if len(text) < 100:
    raise RuntimeError("추출된 텍스트가 너무 짧습니다.")

# 6. 텍스트 파일 저장
output = Path("data/guardduty.txt")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(text, encoding="utf-8")

print("GuardDuty 문서 저장 완료!")
print(f"저장 위치: {output}")
print(f"추출한 글자 수: {len(text):,}")
