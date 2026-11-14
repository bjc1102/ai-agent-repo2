"""파일 저장·해시·시간·실행 식별자를 위한 공통 함수."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo



def now_iso() -> str:
    """한국 시간대가 포함된 현재 시각을 반환."""
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def new_id(prefix: str) -> str:
    """기존 결과와 겹치지 않는 실행 식별자를 생성."""
    return f"{prefix}_{datetime.now(ZoneInfo('Asia/Seoul')):%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"


def sha256_file(path: Path) -> str:
    """파일의 원본 바이트 해시를 계산."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value: object) -> str:
    """JSON 구조의 고정 직렬화 해시를 계산."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """비표준 숫자를 거부하고 같은 폴더에서 원자적으로 JSON을 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)
