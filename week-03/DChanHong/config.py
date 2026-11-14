"""확정된 모델·검색·호출 설정과 프로젝트 경로."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data/golden/golden_v1.json"
CHUNKS_PATH = BASE_DIR / "data/rag/chunks.json"
INDEX_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "dongwon_20261030_v1"
ANSWER_MODEL = "gpt-5.4-mini-2026-03-17"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
TOP_K_PER_REVISION = 3
MAX_OUTPUT_TOKENS = 2000
EVALUATION_MODEL = "gpt-5.4-2026-03-05"
EVALUATION_EMBEDDING_MODEL = "text-embedding-3-large"
EVALUATION_EMBEDDING_DIMENSIONS = 3072
EVALUATION_MAX_OUTPUT_TOKENS = 8192
API_TIMEOUT = 120.0
API_MAX_RETRIES = 2
EXPECTED_VERSIONS = {
    "ragas": "0.4.3", "openai": "2.54.0", "instructor": "1.17.0",
    "chromadb": "1.5.9", "pydantic": "2.14.3", "python-dotenv": "1.2.4",
    "langchain-community": "0.4.1",
}

# USD / 100만 토큰. 실제 usage가 없으면 이 표로 사용량을 추정하지 않습니다.
PRICE_SNAPSHOT = {
    "checked_at": "2026-11-14T16:34:32+09:00",
    "currency": "USD",
    "unit": "per_million_tokens",
    "conditions": "OpenAI standard/default tier, text, input <=272000; no regional uplift",
    "models": {
        ANSWER_MODEL: {"input": 0.75, "cached_input": 0.075, "output": 4.5,
            "source": "https://developers.openai.com/api/docs/models/gpt-5.4-mini"},
        EVALUATION_MODEL: {"input": 2.5, "cached_input": 0.25, "output": 15.0,
            "source": "https://developers.openai.com/api/docs/models/gpt-5.4"},
        EMBEDDING_MODEL: {"input": 0.02, "cached_input": None, "output": None,
            "source": "https://developers.openai.com/api/docs/models/text-embedding-3-small"},
        EVALUATION_EMBEDDING_MODEL: {"input": 0.13, "cached_input": None, "output": None,
            "source": "https://developers.openai.com/api/docs/models/text-embedding-3-large"},
    },
}
