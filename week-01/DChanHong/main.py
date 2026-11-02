# 질문과 응답을 JSON 형식으로 저장하는 기본 모듈
import json

# 실행 시각을 기록하는 기본 모듈
from datetime import datetime

# 설치된 SDK 버전을 확인하는 함수
from importlib.metadata import version

# 결과 파일과 폴더의 경로를 다루는 기본 모듈
from pathlib import Path

# 한국 시간대를 지정하는 기본 모듈
from zoneinfo import ZoneInfo

# OpenAI API 호출 과정에서 발생하는 오류를 처리하는 클래스
from openai import APIError

# 별도 파일에서 생성한 OpenAI 클라이언트 객체
from openai_client import client


# 실험 전체에서 유지할 모델과 지시문
MODEL = "gpt-6-luna"

INSTRUCTIONS = """당신은 메이플스토리의 구매 문의를 돕는 가상의 고객센터 상담 챗봇입니다.
상담 시점은 블랙 큐브가 캐시샵에서 판매되던 2024년 1월 10일입니다.
고객 문의에 한국어로 답변하세요."""

# 모든 호출에 동일하게 적용할 요청 설정
REQUEST_SETTINGS = {
    "model": MODEL,
    "instructions": INSTRUCTIONS,
    "reasoning": {"effort": "none"},
    "max_output_tokens": 2000,
    "store": False,
}

# 한국 시간으로 실행 시각을 기록한다.
KST = ZoneInfo("Asia/Seoul")


# 누적된 기록을 JSON 파일에 저장한다.
def save_results(file_path, data):
    # 임시 파일을 완성한 뒤 교체해 기존 기록의 손상 가능성을 줄인다.
    temporary_path = file_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(file_path)


# 콘솔 입력, API 호출, 응답 출력과 기록을 진행한다.
def main():
    # 현재 파일을 기준으로 결과 폴더를 만들며, 이미 있으면 유지한다.
    result_dir = Path(__file__).resolve().parent / "results" / "01_input_detail"
    result_dir.mkdir(parents=True, exist_ok=True)

    # 실행마다 새로운 결과 파일을 만든다.
    started_at = datetime.now(KST)
    file_path = result_dir / f"{started_at:%Y%m%d_%H%M%S_%f}.json"

    # 자동 재시도를 끄고 요청당 제한 시간을 60초로 설정한다.
    api_client = client.with_options(max_retries=0, timeout=60.0)

    # 이 데이터는 파일 저장에만 사용하며 모델에 대화 이력으로 보내지 않는다.
    data = {
        "experiment": "01_input_detail",
        "started_at": started_at.isoformat(),
        "sdk_version": version("openai"),
        "request_settings": REQUEST_SETTINGS,
        "client_settings": {"max_retries": 0, "timeout_seconds": 60},
        "records": [],
    }
    save_results(file_path, data)

    print(f"모델: {MODEL}")
    print(f"저장 위치: {file_path}")
    print("질문마다 독립적으로 호출합니다. 종료하려면 /exit를 입력하세요.")

    # 사용자가 종료할 때까지 질문을 입력받는다.
    while True:
        try:
            question = input("\n질문 > ").strip()
        # 입력 중 Ctrl+C 또는 입력 종료가 발생하면 프로그램을 끝낸다.
        except (KeyboardInterrupt, EOFError):
            print("\n종료합니다.")
            break

        # 종료 명령이면 API를 호출하지 않는다.
        if question == "/exit":
            break

        # 빈 입력이면 다시 질문을 입력받는다.
        if not question:
            continue

        # 같은 질문을 몇 번 시도했는지 기록한다. 오류도 시도 횟수에 포함한다.
        attempt = 1 + sum(
            record["input"] == question for record in data["records"]
        )

        record = {
            "sequence": len(data["records"]) + 1,
            "requested_at": datetime.now(KST).isoformat(),
            "input": question,
            "attempt_for_same_input": attempt,
        }

        try:
            # 고정 설정과 현재 질문만 보낸다.
            # 이전 질문·응답, previous_response_id, conversation은 보내지 않는다.
            response = api_client.responses.create(
                **REQUEST_SETTINGS,
                input=question,
            )

            # 답변과 원본 응답을 함께 저장한다.
            # 원본에는 모델명, 사용 토큰, 종료 상태와 불완전 응답 정보 등이 포함된다.
            record["status"] = response.status
            record["answer"] = response.output_text
            record["response"] = response.model_dump(mode="json")

            print("\n답변 >")
            print(response.output_text or "(텍스트 답변 없음)")

            # 정상 완료가 아니어도 응답을 버리지 않고 기록한다.
            if response.status != "completed":
                print(f"\n응답 상태: {response.status}")

        # API 오류도 기록하고 다음 질문을 계속 받을 수 있게 한다.
        except APIError as error:
            record["status"] = "api_error"
            record["error"] = {
                "type": type(error).__name__,
                "status_code": getattr(error, "status_code", None),
                "code": getattr(error, "code", None),
                "request_id": getattr(error, "request_id", None),
            }

            # 오류 메시지에 키가 포함될 가능성을 피해 종류와 상태만 출력한다.
            print(
                f"\nAPI 오류: {record['error']['type']} "
                f"(HTTP {record['error']['status_code']})"
            )

        # API 응답을 기다리다가 중단한 경우에도 해당 시도를 기록한다.
        except KeyboardInterrupt:
            record["status"] = "interrupted"
            print("\n요청 대기를 중단했습니다.")

        # 호출 한 번이 끝날 때마다 결과 파일을 갱신한다.
        data["records"].append(record)
        save_results(file_path, data)
        print(f"저장 완료: 같은 질문의 {attempt}번째 시도")

        # 호출 중 사용자가 중단했다면 기록을 저장한 뒤 종료한다.
        if record["status"] == "interrupted":
            break


# 이 파일을 직접 실행했을 때만 콘솔 프로그램을 시작한다.
if __name__ == "__main__":
    main()