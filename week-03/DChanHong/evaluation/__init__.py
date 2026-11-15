"""저장된 답변의 Ragas·규칙·Judge 자동 평가."""

import os

# 평가 모듈을 직접 import해도 프레임워크 분석 이벤트를 전송하지 않습니다.
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
