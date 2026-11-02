# 질문, 응답, 실험 설정을 JSON 파일로 저장한다.
import json

# 실행 시각과 요청 시각을 기록한다.
from datetime import datetime

# 실험에 사용한 라이브러리 버전을 기록한다.
from importlib.metadata import version

# 결과 파일과 폴더의 경로를 다룬다.
from pathlib import Path

# 필드에서 사용할 수 있는 문자열을 제한한다.
from typing import Literal

# 기록 시각을 한국 시간으로 맞춘다.
from zoneinfo import ZoneInfo

# API 호출 오류를 처리한다.
from openai import APIError

# 출력 구조를 정의하고, 반환된 JSON을 검증한다.
from pydantic import BaseModel, ConfigDict, ValidationError

# 기존 파일에서 생성한 OpenAI 클라이언트를 사용한다.
from openai_client import client

MODEL = "gpt-6-luna"
# 1차 진행 테스트
# QUESTION = "10분 전에 블랙 큐브를 결제했는데 아이템이 안 들어왔어요."
QUESTION = "좀전에 큐브 샀는데 템이 없는데?"
REPEATS = 3
KST = ZoneInfo("Asia/Seoul")


# 모델이 반환할 네 가지 필드를 정의하는 Pydantic 스키마이다.
class CustomerResponse(BaseModel):
    #정의하지 않은 추가 필드는 혀용하지 않는다.
    model_config = ConfigDict(extra="forbid")

    request_type: Literal[
        "아이템_미지급",
        "결제_실패",
        "중복_결제",
        "환불_요청",
        "기타",
    ]
    urgency: Literal["낮음", "보통", "높음"]
    customer_reply: str
    next_action: Literal[
        "추가_정보_요청",
        "확인_방법_안내",
        "상담원_연결",
    ]

# 두 실험의 모든 조건에서 동일하게 사용할 지시문이다.

INSTRUCTIONS = """
당신은 메이플스토리의 구매 문의를 돕는 가상의 고객센터 상담 챗봇입니다.
상담 시점은 2024년 1월 10일입니다.
아래 기준은 실험용 가상 고객센터의 기준이며 실제 운영 정책이 아닙니다.
현재 고객 문의만 보고 제공된 JSON 스키마에 맞게 답변하세요.

[request_type]
- 아이템_미지급: 결제했다고 했지만 아이템을 받지 못한 경우.
- 결제_실패: 결제가 거절되거나 오류로 완료되지 않은 경우.
- 중복_결제: 의도한 횟수보다 여러 번 결제되었다고 한 경우.
- 환불_요청: 결제 취소나 환불을 명시적으로 요청한 경우.
- 기타: 위 유형에 해당하지 않거나 정보가 부족해 구분하기 어려운 경우.
여러 문제가 있으면 명시적인 환불 요청을 우선합니다.
그 외에는 고객이 주된 문제로 말한 유형을 선택합니다.

[urgency]
- 높음: 계정 도용이나 계속되는 무단 결제 등 추가 피해 가능성이 명시된 경우.
- 보통: 결제 실패, 아이템 미지급, 중복 결제, 환불 등
  발생한 문제의 확인이나 처리가 필요한 경우.
- 낮음: 현재 발생한 피해 없이 이용 방법이나 정책을 질문하는 경우.
"빨리", "당장" 같은 표현이나 화난 말투만으로 긴급도를 높이지 마세요.
고객이 말하지 않은 피해나 상황을 추측하지 마세요.
정보가 부족해 긴급도를 판단하기 어려우면 보통을 선택하세요.

[next_action]
- 추가_정보_요청: 문의를 이해하거나 다음 안내를 결정하는 데
  필요한 핵심 정보가 부족한 경우.
- 확인_방법_안내: 고객이 직접 해볼 수 있는 확인 방법이나
  문의 절차를 안내할 수 있는 경우.
- 상담원_연결: 추가 피해 우려가 있거나, 기본 확인을 마쳤는데도
  해결되지 않았거나, 실제 결제 조회 또는 환불 처리가 필요한 경우.
여러 기준에 해당하면 다음 순서로 판단하세요.
1. 추가 피해 우려나 담당자의 처리가 필요하면 상담원_연결.
2. 그 외에 안내를 결정할 핵심 정보가 부족하면 추가_정보_요청.
3. 나머지는 확인_방법_안내.

[customer_reply]
- 한국어 존댓말로 1~2문장만 작성하세요.
- 고객 상황에 대한 짧은 반응과 다음 행동 안내를 포함하세요.
- 정보가 부족하면 필요한 내용을 질문하세요.
- 이미 제공한 정보는 다시 묻지 마세요.
- 확인되지 않은 원인, 처리 시간, 환불 가능 여부를 단정하지 마세요.
- next_action으로 선택한 조치와 답변 내용이 일치해야 합니다.
- 실제 조회, 환불, 상담원 연결을 완료했다고 말하지 마세요.
- 상담원_연결을 선택했다면 고객센터 문의 방법을 안내하세요.
- 인사말이나 문장 순서는 자유롭게 구성하세요.
""".strip()

# Pydantic 스키마를 JSON Schema로 변환해 API에도 전달한다.
COMMON_SETTINGS = {
    "model": MODEL,
    "instructions": INSTRUCTIONS,
    "reasoning": {"effort": "none"},
    "max_output_tokens": 2000,
    "store": False,
    "text": {
        "format": {
            "type": "json_schema",
            "name": "customer_response",
            "strict": True,
            "schema": CustomerResponse.model_json_schema(),
        }
    },
}

# 실험마다 하나의 설정만 변경한다.
EXPERIMENTS = [
    {
        "name": "02_temperature",
        "variable": "temperature",
        "values": [0.0, 0.7, 1.3],
        "fixed": {"top_p": 1.0},
    },
    {
        "name": "03_top_p",
        "variable": "top_p",
        "values": [0.1, 0.5, 1.0],
        "fixed": {"temperature": 1.0},
    },
]

# 현재까지의 실험 기록을 저장한다.
def save_results(file_path, data):
    # 임시 파일을 완성한 뒤 교체해 저장 중 손상 가능성을 줄인다.
    temporary_path = file_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(file_path)

# 두 실험을 순서대로 실행하고 호출마다 결과를 저장한다.
def main():
    # 자동 재시도를 끄고, 요청당 제한 시간을 60초로 유지한다.
    api_client = client.with_options(max_retries=0, timeout=60.0)
    base_dir = Path(__file__).resolve().parent / "results"

    for experiment in EXPERIMENTS:
        started_at = datetime.now(KST)
        result_dir = base_dir / experiment["name"]
        result_dir.mkdir(parents=True, exist_ok=True)
        file_path = result_dir / f"{started_at:%Y%m%d_%H%M%S_%f}.json"

        # 이 기록은 파일 저장용이며 모델에 대화 이력으로 보내지 않는다.
        data = {
            "experiment": experiment["name"],
            "started_at": started_at.isoformat(),
            "sdk_version": version("openai"),
            "pydantic_version": version("pydantic"),
            "input": QUESTION,
            "common_settings": COMMON_SETTINGS,
            "client_settings": {
                "max_retries": 0,
                "timeout_seconds": 60,
            },
            "plan": experiment,
            "planned_attempts_per_condition": REPEATS,
            "actual_attempts": 0,
            "records": [],
        }
        save_results(file_path, data)
        print(f"\n실험: {experiment['name']}")
        print(f"저장 위치: {file_path}")

        for value in experiment["values"]:
            # 변경할 설정 하나와 고정할 설정 하나를 합친다.
            sampling_settings = {
                **experiment["fixed"],
                experiment["variable"]: value,
            }

            for attempt in range(1, REPEATS + 1):
                record = {
                    "sequence": len(data["records"]) + 1,
                    "requested_at": datetime.now(KST).isoformat(),
                    "input": QUESTION,
                    "sampling_settings": sampling_settings,
                    "attempt_for_condition": attempt,
                    "status": "started",
                    "validation_status": "not_checked",
                    "parsed": None,
                }

                # 요청 시작을 먼저 기록해 실행 도중 종료되어도 흔적을 남긴다.
                data["records"].append(record)
                data["actual_attempts"] = len(data["records"])
                save_results(file_path, data)

                print(f"\n설정: {sampling_settings} / {attempt}회차")

                try:
                    # 현재 질문과 고정 지시문만 보낸다.
                    # 이전 질문, 응답, conversation, previous_response_id는 보내지 않는다.
                    response = api_client.responses.create(
                        **COMMON_SETTINGS,
                        **sampling_settings,
                        input=QUESTION,
                    )

                    # 검증 전에 답변과 원본 응답을 먼저 보존한다.
                    record["status"] = response.status
                    record["answer"] = response.output_text
                    record["response"] = response.model_dump(mode="json")
                    save_results(file_path, data)

                    # 거절 응답은 일반 JSON 답변과 구분한다.
                    refusals = [
                        part.get("refusal", "")
                        for item in record["response"].get("output", [])
                        for part in item.get("content", [])
                        if part.get("type") == "refusal"
                    ]

                    # 불완전 응답은 원본을 보존하고 성공으로 처리하지 않는다.
                    if response.status != "completed":
                        record["validation_status"] = "not_completed"

                    # 모델이 응답을 거절했다면 거절 내용을 기록한다.
                    elif refusals:
                        record["validation_status"] = "refusal"
                        record["refusals"] = refusals

                    # 정상 완료된 JSON은 Pydantic으로 필드와 허용 값을 검증한다.
                    else:
                        parsed = CustomerResponse.model_validate_json(
                            response.output_text
                        )
                        record["parsed"] = parsed.model_dump(mode="json")
                        record["validation_status"] = "passed"

                    print(response.output_text or "(텍스트 답변 없음)")

                # 형식 검증에 실패해도 이미 저장한 원본 응답을 유지한다.
                except ValidationError as error:
                    record["validation_status"] = "failed"
                    record["validation_errors"] = error.errors(
                        include_url=False,
                        include_input=False,
                        include_context=False,
                    )
                    print("Pydantic 검증 실패: 원본 응답을 기록했습니다.")

                # API 오류도 시도 횟수에 포함하고 다음 호출을 진행한다.
                except APIError as error:
                    record["status"] = "api_error"
                    record["error"] = {
                        "type": type(error).__name__,
                        "status_code": getattr(error, "status_code", None),
                        "code": getattr(error, "code", None),
                        "param": getattr(error, "param", None),
                        "request_id": getattr(error, "request_id", None),
                    }
                    # 키가 포함될 가능성이 있는 요청 헤더나 오류 본문은 출력하지 않는다.
                    print(f"API 오류: {record['error']}")

                # 호출 중 Ctrl+C를 누르면 해당 시도를 기록하고 종료한다.
                except KeyboardInterrupt:
                    record["status"] = "interrupted"
                    print("\n중단한 요청을 기록합니다.")

                # 성공, 오류, 중단 여부와 관계없이 현재 기록을 저장한다.
                finally:
                    record["finished_at"] = datetime.now(KST).isoformat()
                    save_results(file_path, data)

                print(
                    f"저장 완료: 상태={record['status']}, "
                    f"검증={record['validation_status']}"
                )

                # 중단한 경우 다음 실험까지 진행하지 않고 프로그램을 끝낸다.
                if record["status"] == "interrupted":
                    return

        print(f"\n실험 종료: 실제 시도 {data['actual_attempts']}회")

    print("\n전체 실험이 끝났습니다.")


# 이 파일을 직접 실행했을 때 실험을 시작한다.
if __name__ == "__main__":
    main()    