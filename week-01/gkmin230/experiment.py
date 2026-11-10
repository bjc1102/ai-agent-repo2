"""Run six independent Responses API requests; never synthesize model output."""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

MODEL = "gpt-5.4-mini-2026-03-17"
LOG = "GET /.env HTTP/1.1\nUser-Agent: python-requests/2.31.0\nHTTP Status: 403"
INSTRUCTIONS = {
    "A": "다음 HTTP 로그를 분석해줘.",
    "B": "다음 HTTP 로그를 공격 의심 여부, 판단 근거, 추가 확인 사항으로 구분해 분석해줘.",
}
SETTINGS = {"max_output_tokens": 1600, "reasoning": {"effort": "none"}, "store": False}
DOCS = {
    "checked_date_kst": "2026-10-31",
    "model": "https://developers.openai.com/api/docs/models/gpt-5.4-mini",
    "api": "https://developers.openai.com/api/reference/python/resources/responses/methods/create",
    "sdk": "https://developers.openai.com/api/reference/python",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def redact(value):
    """Keep responses intact except credentials that an error might echo."""
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            value = value.replace(key, "[REDACTED]")
        return re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED]", value)
    return value


def save(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(redact(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("기존 결과를 보존합니다. --output으로 새 파일 경로를 지정하세요.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        sdk_version = version("openai")
    except PackageNotFoundError:
        sdk_version = None
    setup_error = None
    client = None
    try:
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다. API 요청을 전송하지 않았습니다.")
        from openai import OpenAI
        # Explicit official endpoint: environment overrides cannot redirect the key.
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                        base_url="https://api.openai.com/v1", max_retries=0, timeout=90.0)
    except Exception as exc:
        setup_error = {"type": type(exc).__name__, "message": str(exc)}
    data = {
        "experiment": "프롬프트 구체성에 따른 보안 로그 분석 결과 비교",
        "started_at_utc": now(), "model_requested": MODEL,
        "sdk": {"name": "openai", "version": sdk_version, "python": sys.version.split()[0]},
        "api": "POST https://api.openai.com/v1/responses", "input_log": LOG,
        "instructions": INSTRUCTIONS, "request_settings": SETTINGS,
        "temperature_sent": False, "repetitions_per_condition": 3,
        "planned_calls": 6, "order": "A1, B1, A2, B2, A3, B3",
        "independence": "Each request contains only its instruction and the same log; no conversation or previous_response_id.",
        "max_retries": 0, "official_documentation": DOCS,
        "setup_error": setup_error, "runs": [],
    }
    save(args.output, data)
    for repeat in range(1, 4):
        for condition, instruction in INSTRUCTIONS.items():
            prompt = instruction + "\n\n" + LOG
            request = {"model": MODEL, "input": prompt, **SETTINGS}
            run = {"condition": condition, "repeat": repeat, "prompt": prompt,
                   "request": request, "started_at_utc": now(),
                   "api_call_attempted": False, "success": False,
                   "response": None, "output_text": None, "error": None}
            start = time.monotonic()
            if setup_error:
                run.update(status="not_attempted", error=setup_error)
            else:
                try:
                    run["api_call_attempted"] = True
                    response = client.responses.create(**request)
                    run["response"] = response.model_dump(mode="json")
                    run["output_text"] = response.output_text
                    run["request_id"] = getattr(response, "_request_id", None)
                    run["model_returned"] = response.model
                    run["status"] = response.status
                    run["success"] = (response.status == "completed" and
                                      bool(response.output_text.strip()) and not response.error)
                    if not run["success"]:
                        run["error"] = {"type": "UnsuccessfulResponse", "message": "응답 미완료 또는 텍스트 없음",
                                        "api_error": run["response"].get("error"),
                                        "incomplete_details": run["response"].get("incomplete_details")}
                except Exception as exc:
                    run["status"] = "error"
                    run["error"] = {"type": type(exc).__name__, "message": str(exc),
                                    "http_status": getattr(exc, "status_code", None),
                                    "request_id": getattr(exc, "request_id", None),
                                    "body": getattr(exc, "body", None)}
            run["elapsed_seconds"] = round(time.monotonic() - start, 3)
            data["runs"].append(run)
            save(args.output, data)
            print(f"{condition}{repeat}: {run['status']} (success={run['success']})", flush=True)
    if client:
        client.close()
    data["finished_at_utc"] = now()
    data["summary"] = {
        "planned_calls": 6,
        "api_call_attempts": sum(r["api_call_attempted"] for r in data["runs"]),
        "successful_calls": sum(r["success"] for r in data["runs"]),
        "errors": sum(r["error"] is not None for r in data["runs"]),
        "not_attempted": sum(r["status"] == "not_attempted" for r in data["runs"]),
        "by_condition": {c: {"planned": 3, "successful": sum(r["success"] for r in data["runs"] if r["condition"] == c)} for c in INSTRUCTIONS},
    }
    save(args.output, data)
    print(json.dumps(data["summary"], ensure_ascii=False))
    return 0 if data["summary"]["successful_calls"] == 6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
