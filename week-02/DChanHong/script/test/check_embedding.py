import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY가 없습니다. .env 설정을 확인해주세요.")

MODEL = "text-embedding-3-small"
text = "이 문서는 공모가격과 자금 사용 계획을 설명합니다."

# OpenAI()는 환경변수의 API 키를 사용합니다.
client = OpenAI(timeout=30.0, max_retries=0)

response = client.embeddings.create(
    model=MODEL,
    input=text,
    encoding_format="float",
)

embedding = response.data[0].embedding

print(f"모델: {response.model}")
print(f"입력 문장: {text}")
print(f"벡터 차원: {len(embedding)}")
print(f"벡터 앞 5개 값: {embedding[:5]}")
print(f"사용 토큰 수: {response.usage.total_tokens}")

if len(embedding) != 1536:
    raise SystemExit("예상한 1,536차원과 다릅니다.")

print("임베딩 테스트 성공")