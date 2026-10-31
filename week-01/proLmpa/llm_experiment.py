import argparse
import datetime
import json
import re
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 8000
BASE_EFFORT = "high"
RESULTS = Path("results.json1")
REPORT = Path("report.md")

client = anthropic.Anthropic(max_retries=0) # SDK 자동 재시도 OFF (max_retries=0)

# --- Inputs --- 
E1_SYSTEM = "당신은 계산 문제를 푸는 조수다."
E1_USER = (
    "1 이상 200 이하의 정수 중 3 또는 5로 나누어떨어지지만 15로는 나누어떨어지지 않는 "
    "수의 합을 구하라. 마지막 줄은 반드시 '정답: <숫자>' 형식으로 끝내라."
)
E1_ANSWER = 8003  # 3의 배수 합 6633 + 5의 배수 합 4100 - 15의 배수 합 1365 x 2
 
E2_SYSTEM = "당신은 영화 리뷰 감성 분류기다."
E2_REVIEWS = [
    "배우 연기는 좋았는데 결말이 너무 허무했다.",
    "두 시간이 순식간에 지나갔다. 또 보고 싶다.",
    "그냥 평범했다. 시간 때우기용.",
    "음악 빼고는 기억에 남는 게 없다.",
    "기대 안 했는데 의외로 괜찮았다.",
]
E2_LABELS = {"긍정", "부정", "중립"}
E2_INSTRUCTION = "각 리뷰의 감성을 긍정, 부정, 중립 중 하나로 분류하라. 리뷰당 한 줄씩 라벨만 출력하라."
E2_FEWSHOT = (
    "예시 입력:\n"
    "1. 스토리가 탄탄하고 몰입감이 대단했다.\n"
    "2. 돈이 아까웠다. 중간에 나오고 싶었다.\n"
    "3. 볼 만은 한데 다시 볼 정도는 아니다.\n"
    "예시 출력:\n긍정\n부정\n중립\n\n"
)
E2_TARGET = "입력:\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(E2_REVIEWS, 1))
 
E3_SYSTEM = "당신은 문서에서 정보를 추출하는 조수다."
E3_ARTICLE = (
    "2026년 3월 12일, 서울시는 한강 공원 세 곳에 공공 와이파이를 추가 설치한다고 발표했다. "
    "사업은 디지털정책과가 맡으며 총예산은 4억 5천만 원이다. 설치는 같은 해 6월 말까지 마칠 예정이다."
)
E3_USER = (
    "다음 문단에서 date(발표일, YYYY-MM-DD), organization(담당 부서), "
    "budget_krw(총예산, 원 단위 정수), deadline(완료 예정 시점)을 추출해 JSON 객체로만 답하라.\n\n"
    + E3_ARTICLE
)
E3_SCHEMA = {
    "type": "object",
    "properties": {
        "date": {"type": "string"},
        "organization": {"type": "string"},
        "budget_krw": {"type": "integer"},
        "deadline": {"type": "string"},
    },
    "required": ["date", "organization", "budget_krw", "deadline"],
    "additionalProperties": False,
}
 
E4_SYSTEM = "당신은 친절한 조수다."
E4_USER = "LLM이 무엇인지 한 문장으로 설명하라."

E5_SYSTEM = "당신은 계산 문제를 푸는 조수다."
E5_USER = (
    "100 이상 200 이하의 정수 중 소수(prime number)인 수를 모두 찾아 합을 구하라. "
    "마지막 줄은 반드시 '정답: <숫자>' 형식으로 끝내라."
)
E5_ANSWER = 3167  # 소수: 101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199

E6_SYSTEM = "당신은 친절한 조수다."
E6_USER = "LLM의 주요 특징 3가지를 번호를 붙여 한 문장씩 설명하라."

# --- Check Functions ---
def check_e1(text):
    m = re.findall(r"정답:\s*([\d,]+)", text)
    if not m:
        return {"format_ok": False, "correct": False}
    value = int(m[-1].replace(",", ""))
    return {"format_ok": True, "answer": value, "correct": value == E1_ANSWER}


def check_e2(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    ok = len(lines) == len(E2_REVIEWS) and all(l in E2_LABELS for l in lines)
    return {"format_ok": ok, "labels": "/".join(lines)}


def check_e3(text):
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"json_ok": False}
    keys_ok = set(data) == set(E3_SCHEMA["required"]) if isinstance(data, dict) else False
    return {
        "json_ok": True,
        "keys_ok": keys_ok,
        "budget_ok": isinstance(data, dict) and data.get("budget_krw") == 450_000_000,
    }


def check_e5(text):
    m = re.findall(r"정답:\s*([\d,]+)", text)
    if not m:
        return {"format_ok": False, "correct": False}
    value = int(m[-1].replace(",", ""))
    return {"format_ok": True, "answer": value, "correct": value == E5_ANSWER}


def check_e6(text):
    items = re.findall(r"^\d+\.", text, re.MULTILINE)
    return {"item_count": len(items)}


def check_none(text):
    return {}

# --- Experiment Parameters ---
EXPERIMENTS = {
    "E1": {
        "title": "effort low vs high",
        "changed": "output_config.effort (low / high)",
        "fixed": f"model={MODEL}, system, user(계산 문제), max_tokens={MAX_TOKENS}, 대화 이력 없음",
        "check": check_e1,
        "conds": [
            ("effort=low", E1_SYSTEM, E1_USER, {"effort": "low"}, {}),
            ("effort=high", E1_SYSTEM, E1_USER, {"effort": "high"}, {}),
        ],
    },
    "E2": {
        "title": "zero-shot vs few-shot",
        "changed": "예시 3개 포함 여부",
        "fixed": f"model={MODEL}, system, 지시문, 분류 대상 리뷰 5개, effort={BASE_EFFORT}, max_tokens={MAX_TOKENS}",
        "check": check_e2,
        "conds": [
            ("zero-shot", E2_SYSTEM, f"{E2_INSTRUCTION}\n\n{E2_TARGET}", {"effort": BASE_EFFORT}, {}),
            ("few-shot", E2_SYSTEM, f"{E2_INSTRUCTION}\n\n{E2_FEWSHOT}{E2_TARGET}", {"effort": BASE_EFFORT}, {}),
        ],
    },
    "E3": {
        "title": "프롬프트 JSON 지시 vs 구조화 출력",
        "changed": "output_config.format(JSON 스키마) 지정 여부",
        "fixed": f"model={MODEL}, system, user(같은 지시문과 문단), effort={BASE_EFFORT}, max_tokens={MAX_TOKENS}",
        "check": check_e3,
        "conds": [
            ("prompt-only", E3_SYSTEM, E3_USER, {"effort": BASE_EFFORT}, {}),
            ("json_schema", E3_SYSTEM, E3_USER,
             {"effort": BASE_EFFORT, "format": {"type": "json_schema", "schema": E3_SCHEMA}}, {}),
        ],
    },
    "E4": {
        "title": "temperature 지정 시 동작 확인",
        "changed": "temperature 파라미터 지정 여부 (미지정 / 0.7)",
        "fixed": f"model={MODEL}, system, user, effort={BASE_EFFORT}, max_tokens={MAX_TOKENS}",
        "check": check_none,
        "conds": [
            ("no-temperature", E4_SYSTEM, E4_USER, {"effort": BASE_EFFORT}, {}),
            ("temperature=0.7", E4_SYSTEM, E4_USER, {"effort": BASE_EFFORT}, {"temperature": 0.7}),
        ],
    },
    "E5": {
        "title": "effort 전 단계 비교 (low/medium/high/xhigh/max)",
        "changed": "output_config.effort (low/medium/high/xhigh/max)",
        "fixed": f"model={MODEL}, system, user(소수 합산), max_tokens={MAX_TOKENS}",
        "check": check_e5,
        "conds": [
            ("effort=low",    E5_SYSTEM, E5_USER, {"effort": "low"},    {}),
            ("effort=medium", E5_SYSTEM, E5_USER, {"effort": "medium"}, {}),
            ("effort=high",   E5_SYSTEM, E5_USER, {"effort": "high"},   {}),
            ("effort=xhigh",  E5_SYSTEM, E5_USER, {"effort": "xhigh"},  {}),
            ("effort=max",    E5_SYSTEM, E5_USER, {"effort": "max"},    {}),
        ],
    },
    "E6": {
        "title": "stop_sequences 유무 비교",
        "changed": "stop_sequences (없음 / ['2.'])",
        "fixed": f"model={MODEL}, system, user(번호 목록 3개), max_tokens={MAX_TOKENS}, output_config 없음",
        "check": check_e6,
        "conds": [
            ("no-stop",   E6_SYSTEM, E6_USER, {}, {}),
            ("stop-at-2", E6_SYSTEM, E6_USER, {}, {"stop_sequences": ["2."]}),
        ],
    },
    "E7": {
        "title": "thinking.type 비교 (disabled/adaptive/enabled)",
        "changed": "thinking.type (disabled / adaptive / enabled)",
        "fixed": f"model={MODEL}, system, user(감성 분류), effort={BASE_EFFORT}, max_tokens={MAX_TOKENS}",
        "check": check_e2,
        "conds": [
            ("thinking=disabled", E2_SYSTEM, f"{E2_INSTRUCTION}\n\n{E2_FEWSHOT}{E2_TARGET}",
             {"effort": BASE_EFFORT}, {"thinking": {"type": "disabled"}}),
            ("thinking=adaptive", E2_SYSTEM, f"{E2_INSTRUCTION}\n\n{E2_FEWSHOT}{E2_TARGET}",
             {"effort": BASE_EFFORT}, {"thinking": {"type": "adaptive"}}),
            ("thinking=enabled",  E2_SYSTEM, f"{E2_INSTRUCTION}\n\n{E2_FEWSHOT}{E2_TARGET}",
             {"effort": BASE_EFFORT}, {"thinking": {"type": "enabled", "budget_tokens": 2000}}),
        ],
    },
}

# --- Experiment Execution ---
def call(exp, cond, run, system, user, output_config, extra):
    req = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        **extra,
    }
    extra_body = {"output_config": output_config} if output_config else {}
    rec = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "exp": exp, "cond": cond, "run": run,
        "sdk_version": anthropic.__version__,
        "request": {**req, **({"output_config": output_config} if output_config else {})},
    }
    t0 = time.perf_counter()
    try:
        resp = client.messages.create(**req, extra_body=extra_body)
        text = "".join(b.text for b in resp.content if b.type == "text")
        rec.update(
            ok=True,
            model_returned=resp.model,
            stop_reason=resp.stop_reason,
            block_types=[b.type for b in resp.content],
            usage=resp.usage.model_dump(),
            text=text,
            check=EXPERIMENTS[exp]["check"](text),
        )
    except anthropic.APIStatusError as e:
        rec.update(ok=False, status=e.status_code, error=str(e))
    except anthropic.APIError as e:
        rec.update(ok=False, status=None, error=f"{type(e).__name__}: {e}")
    except TypeError as e:
        rec.update(ok=False, status=None, error=f"TypeError: {e}")
    rec["latency_s"] = round(time.perf_counter() - t0, 2)
 
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def run_experiment(exp, runs):
    spec = EXPERIMENTS[exp]
    print(f"\n[{exp}] {spec['title']} | 반복 {runs}회")
    for label, system, user, oc, extra in spec["conds"]:
        for i in range(1, runs + 1):
            r = call(exp, label, i, system, user, oc, extra)
            if r["ok"]:
                print(f"  {label:<16} #{i} {r['stop_reason']:<12} "
                      f"out={r['usage'].get('output_tokens')} {r['latency_s']}s {r['check']}")
            else:
                print(f"  {label:<16} #{i} ERROR {r['status']} {r['error'][:120]}")

# --- Reports ---
def cell(s, n=160):
    s = str(s).replace("|", "\\|").replace("\n", " ⏎ ")
    return s if len(s) <= n else s[:n] + "…"
 
 
def build_report():
    if not RESULTS.exists():
        print("results.jsonl이 없습니다.")
        return
    recs = [json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = ["# LLM 호출 실험 결과", ""]
    for exp, spec in EXPERIMENTS.items():
        rows = [r for r in recs if r["exp"] == exp]
        if not rows:
            continue
        sdk = sorted({r["sdk_version"] for r in rows})
        out += [
            f"## {exp}. {spec['title']}", "",
            f"- 모델: {MODEL} / SDK: anthropic {', '.join(sdk)}",
            f"- 바꾼 조건: {spec['changed']}",
            f"- 유지한 조건: {spec['fixed']}",
        ]
        for label, *_ in spec["conds"]:
            n = sum(1 for r in rows if r["cond"] == label)
            out.append(f"- 실제 반복 횟수 ({label}): {n}회")
        out += [
            "",
            "| 조건 | 회차 | stop_reason | 입력/출력 토큰 | 지연(s) | 자동 점검 | 오류 | 응답 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in rows:
            if r["ok"]:
                u = r["usage"]
                out.append(
                    f"| {r['cond']} | {r['run']} | {r['stop_reason']} | "
                    f"{u.get('input_tokens')}/{u.get('output_tokens')} | {r['latency_s']} | "
                    f"{cell(json.dumps(r['check'], ensure_ascii=False), 80)} | - | {cell(r['text'])} |"
                )
            else:
                out.append(
                    f"| {r['cond']} | {r['run']} | - | - | {r['latency_s']} | - | "
                    f"{r['status']}: {cell(r['error'], 200)} | - |"
                )
        if exp == "E2":  # 반복 간 라벨 일치 여부
            out.append("")
            for label, *_ in spec["conds"]:
                sets = {r["check"]["labels"] for r in rows if r["cond"] == label and r["ok"]}
                out.append(f"- {label}: 서로 다른 라벨 조합 {len(sets)}종 → {', '.join(sorted(sets))}")
        if exp == "E7":
            out.append("")
            for label, *_ in spec["conds"]:
                toks = [r["usage"]["output_tokens_details"]["thinking_tokens"]
                        for r in rows if r["cond"] == label and r["ok"]]
                avg = round(sum(toks)/len(toks), 1) if toks else "N/A"
                sets = {r["check"]["labels"] for r in rows if r["cond"] == label and r["ok"]}
                out.append(f"- {label}: thinking_tokens 평균 {avg} | 라벨 조합 {len(sets)}종")
        out += ["", "**관찰 (직접 작성, 3~5줄)**", "", "- ", ""]
    REPORT.write_text("\n".join(out), encoding="utf-8")
    print(f"\nreport.md 작성 완료 ({len(recs)}건 기록 기준)")
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", nargs="+", choices=list(EXPERIMENTS), default=[])
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args()
    if not a.report_only:
        if not a.exp:
            p.error("--exp E1 처럼 실험을 지정하세요")
        for e in a.exp:
            run_experiment(e, a.runs)
    build_report()
 
 
if __name__ == "__main__":
    main()