"""API·파일 처리·검증에 사용하는 공통 기능."""

import os

# 직접 모듈을 불러올 때도 통계 수집을 비활성화합니다.
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
