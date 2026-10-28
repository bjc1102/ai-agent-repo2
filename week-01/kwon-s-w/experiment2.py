# %% 준비: Colab에서 google-genai 설치와 보안 비밀 등록 후 실행
import csv
import json
import random
import time
import uuid
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

# 새 Colab 노트북에서 SDK가 없을 때만 설치합니다. 기존 설치는 변경하지 않습니다.
try:
    version("google-genai")
except Exception as error:
    from importlib.metadata import PackageNotFoundError
    if not isinstance(error, PackageNotFoundError):
        raise
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "google-genai", "google-auth==2.49.0"])

from google import genai
from google.genai import types
from google.colab import userdata

from getpass import getpass
try:
    _api_key = userdata.get("GEMINI_API_KEY")
except (userdata.TimeoutException, userdata.SecretNotFoundError, userdata.NotebookAccessError):
    _api_key = getpass("Gemini API 키 입력 (화면에 표시되지 않음): ")
client = genai.Client(api_key=_api_key)
del _api_key
MODEL = "gemini-3.5-flash-lite"
REPEATS = 3
SDK_VERSION = version("google-genai")

RULES = """고객 문의를 다음 기준으로 분류하세요.
- 배송지를 바꾸려는 요청: category=배송지 변경, department=주문변경팀
- 상품이 도착하지 않거나 배송이 늦어 도착 시점을 묻는 문의: category=배송 지연, department=배송팀
- 구매 취소 후 돈을 돌려받으려는 요청: category=환불 요청, department=환불팀
- 변경할 대상 등 요청 의도가 불명확하면: category=의도 불명확, department=추가확인
고객이 부정한 의도는 선택하지 마세요. 근거가 부족하면 추측하지 마세요.
지정한 JSON 구조로만 응답하세요."""

EXAMPLES = """
분류 예시:
문의: 받는 곳을 사무실로 바꿀 수 있나요?
응답: {"category":"배송지 변경","department":"주문변경팀"}
문의: 도착 예정일이 지났는데 아직 못 받았어요.
응답: {"category":"배송 지연","department":"배송팀"}
문의: 물건을 돌려보내고 결제한 돈을 받고 싶어요.
응답: {"category":"환불 요청","department":"환불팀"}
문의: 주문한 것 좀 처리해주세요.
응답: {"category":"의도 불명확","department":"추가확인"}
"""

CONDITIONS = {"A_예시없음": RULES, "B_예시있음": RULES + EXAMPLES}
TESTS = [
    {"id": "Q1", "input": "어제는 배송이 늦으면 환불하겠다고 했는데, 환불은 하지 않고 그냥 받겠습니다. 오늘 요청하는 건 받을 주소를 회사에서 집으로 바꾸는 것뿐이에요.",
     "expected": {"category": "배송지 변경", "department": "주문변경팀"}},
    {"id": "Q2", "input": "처음에는 주소를 잘못 적어서 배송지를 바꾸려 했지만 그 문제는 이미 해결됐어요. 취소나 환불도 원하지 않습니다. 도착 예정일이 지났는데 지금 상품이 어디에 있고 언제 도착하나요?",
     "expected": {"category": "배송 지연", "department": "배송팀"}},
    {"id": "Q3", "input": "배송지를 바꾸거나 환불해 달라는 뜻은 아니에요. 주문한 내용을 조금 수정하고 싶은데 처리해 주실 수 있나요?",
     "expected": {"category": "의도 불명확", "department": "추가확인"}},
]
SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "category": {"type": "STRING", "enum": ["배송지 변경", "배송 지연", "환불 요청", "의도 불명확"]},
        "department": {"type": "STRING", "enum": ["주문변경팀", "배송팀", "환불팀", "추가확인"]},
    },
    "required": ["category", "department"],
}
# 두 조건에 동일한 설정을 사용합니다.
COMMON_CONFIG = {"response_mime_type": "application/json", "response_schema": SCHEMA,
                 "max_output_tokens": 4096}
# temperature / top_p / thinking은 보내지 않고 모델의 기본 설정을 공통 사용합니다.
print(f"준비 완료 / 모델: {MODEL} / SDK: {SDK_VERSION} / 호출 예정: 18회")

# %% 실험: 이 셀을 다시 실행하면 새로운 18회 호출. 이전 결과는 별도 폴더에 보존.
run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
OUT = Path("week01_results_experiment2") / run_id
OUT.mkdir(parents=True, exist_ok=False)
records = []
plan = {
    "run_id": run_id, "model": MODEL, "sdk": "google-genai", "sdk_version": SDK_VERSION,
    "variable": "분류 예시의 유무만 변경", "conditions": CONDITIONS, "tests": TESTS,
    "common_config": COMMON_CONFIG, "repeats_per_input_condition": REPEATS,
    "history": "매번 독립적인 generate_content 호출. 이전 응답을 전달하지 않음.",
    "unspecified_parameters": "temperature, top_p, thinking 미전송. 제공자 기본값 사용.",
    "execution_order": "각 반복의 6건을 seed=42로 섞음",
    "retry_policy": "실험 코드에서 재시도하지 않음. SDK 내부 HTTP 재시도는 기본 동작 사용.",
}
(OUT / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "requirements.txt").write_text(f"google-genai=={SDK_VERSION}\n", encoding="utf-8")
rng = random.Random(42)

for repetition in range(1, REPEATS + 1):
    jobs = [(condition, test) for condition in CONDITIONS for test in TESTS]
    rng.shuffle(jobs)
    for condition, test in jobs:
        record = {
            "sequence": len(records) + 1, "repetition": repetition,
            "condition": condition, "test_id": test["id"], "model": MODEL,
            "sdk_version": SDK_VERSION, "input": test["input"], "expected": test["expected"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "completed": False, "schema_valid": False, "correct": False,
            "text": "", "parsed": None, "finish_reason": None, "error": None,
        }
        started = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=MODEL, contents=test["input"],
                config=types.GenerateContentConfig(system_instruction=CONDITIONS[condition], **COMMON_CONFIG),
            )
            # 실제 응답 전체를 저장합니다. API 키와 요청 헤더는 포함하지 않습니다.
            record["raw_response"] = json.loads(response.model_dump_json())
            record["model_version"] = getattr(response, "model_version", None)
            candidates = response.candidates or []
            if candidates:
                reason = candidates[0].finish_reason
                record["finish_reason"] = getattr(reason, "value", str(reason))
            record["completed"] = record["finish_reason"] == "STOP"
            record["text"] = response.text or ""
            try:
                parsed = json.loads(record["text"])
                record["parsed"] = parsed
                valid = isinstance(parsed, dict) and set(parsed) == {"category", "department"}
                valid = valid and all(isinstance(parsed[k], str) and parsed[k] in SCHEMA["properties"][k]["enum"]
                                      for k in ("category", "department"))
                record["schema_valid"] = bool(valid)
                record["correct"] = bool(record["completed"] and valid and parsed == test["expected"])
            except (ValueError, TypeError) as error:
                record["validation_error"] = type(error).__name__
        except Exception as error:
            # 예외 전문 대신 유형과 코드만 기록합니다.
            record["error"] = {"type": type(error).__name__, "code": str(getattr(error, "code", ""))}
        record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        records.append(record)
        with (OUT / "responses.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f'{record["sequence"]:02d}/18 | {condition} | {test["id"]} | {repetition}회차 '
              f'| 완료={record["completed"]} | 구조={record["schema_valid"]} | 정답={record["correct"]}')
        print(record["text"] or record["error"] or record["finish_reason"])
        time.sleep(5)

# %% 집계·다운로드: API를 호출하지 않습니다.
summary = []
for condition in CONDITIONS:
    rows = [r for r in records if r["condition"] == condition]
    summary.append({
        "condition": condition, "attempts": len(rows),
        "completed": sum(r["completed"] for r in rows),
        "schema_valid": sum(r["schema_valid"] for r in rows),
        "correct": sum(r["correct"] for r in rows),
        "api_errors": sum(r["error"] is not None for r in rows),
    })
with (OUT / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
print(json.dumps(summary, ensure_ascii=False, indent=2))

lines = ["# 1주차 LLM 호출 실험 — 복합 표현 문의", "", "## 실험 목적",
         "분류 예시를 제공했을 때 고객 문의 분류 결과가 달라지는지 관찰한다.", "",
         "## 실험 방법",
         "- 입력은 실제 고객 자료가 아닌 실험용 가상 문장이다. A와 B에는 동일하게 사용한다.",
         f"- 모델: `{MODEL}` / SDK: `google-genai=={SDK_VERSION}`",
         "- 변경한 조건: 분류 예시의 유무만 변경. B는 A의 지시에 예시 4개만 추가.",
         "- 유지한 조건: 모델, 문의 3개, 분류 기준, JSON Schema, 출력 한도 4096, 기본 생성 설정.",
         "- 대화 이력 없이 독립 호출. 문의·조건별 3회, 총 18회 SDK 호출 시도. 실험 코드의 자동 재시도 없음(SDK 내부 동작은 기본값).",
         "- 기대 결과는 호출 전에 정의. 자세한 지시와 입력은 plan.json에 기록.", "",
         "## 실제 집계", "", "| 조건 | 시도 | 정상 완료 | 구조 통과 | 완료·구조·정답 모두 통과 | API 오류 |",
         "|---|---:|---:|---:|---:|---:|"]
for s in summary:
    lines.append(f'| {s["condition"]} | {s["attempts"]} | {s["completed"]} | {s["schema_valid"]} | {s["correct"]} | {s["api_errors"]} |')
lines += ["", "## 관찰 (실제 결과를 바탕으로 자동 집계)"]
for s in summary:
    lines.append(f'- {s["condition"]}: 총 {s["attempts"]}회 중 정상 완료 {s["completed"]}회, 구조 통과 {s["schema_valid"]}회, 정답 통과 {s["correct"]}회였다.')
a, b = summary
lines += [f'- 예시 있는 조건의 정답 통과 건수와 없는 조건의 차이는 {b["correct"] - a["correct"]}건이다. 원인이나 일반적인 성능 우위를 입증하는 수치는 아니다.',
          '- API 오류와 불완전 응답은 별도로 기록했으며, 성공한 응답처럼 취급하지 않았다.',
          '- 입력이 3개이고 조건별 반복이 3회인 소규모 실험이므로 다른 입력에 일반화하지 않는다. 의도 불명확/추가확인은 정보 부족에 대한 정상 분류이며 호출 실패가 아니다.']

(OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
archive = OUT.parent / (run_id + ".zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(OUT.iterdir()):
        bundle.write(path, arcname=path.name)
print("결과 저장:", archive)
from google.colab import files
files.download(str(archive))
