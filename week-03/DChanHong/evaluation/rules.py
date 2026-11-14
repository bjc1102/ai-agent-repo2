"""안정적으로 연결되는 수치·계산식·인용 라벨만 기계적으로 검사한다."""

import ast
import re
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP

from schemas import GoldenQuestion


ALIASES = {
    "주당 평가가액": ("주당 평가가액", "평가가액"),
    "확정 주당 공모가액": ("확정 주당 공모가액", "확정 공모가액", "확정 공모가", "주당 공모가액"),
    "평가가액 대비 공모가액 차액": ("차액", "금액 차이"),
    "평가가액 대비 공모가액 할인율": ("할인율",),
    "시설자금 3개년 합계": ("시설자금 3개년 합계", "시설자금 합계", "3개년 합계"),
    "건축자금 제외 시설자금": ("건축자금 제외 시설자금", "건축자금 제외", "건축자금을 제외"),
    "100주 청약 시 수수료 제외 청약증거금": ("청약증거금",),
    "일반청약 증거금률": ("일반청약 증거금률", "청약증거금률", "증거금률"),
}
NUMBER = r"-?\d[\d,]*(?:\.\d+)?"
UNITS = r"백만\s*원|원|주|%|퍼센트"


def normalized(text: str) -> str:
    """마크다운 강조와 숫자 구분 쉼표·공백만 비교에서 제거한다."""
    return re.sub(r"\s+", "", text.replace(",", "").replace("**", "")).replace("퍼센트", "%")


def target_lines(rule: dict, response: str) -> list[tuple[str, str]]:
    """기대 수치를 보지 않고 대상·기간이 한 줄에 명시된 후보를 고른다."""
    names = ALIASES.get(rule["target"], (rule["target"],))
    candidates = []
    for line in response.splitlines():
        clean = line.replace("**", "")
        # 부정·예시·가정 문장 속 숫자를 실제 주장 값으로 채택하지 않는다.
        if re.search(r"아니|아닌|아님|아닙|않|라면|가정|예를|예시|추정|가설", clean):
            continue
        matches = [(clean.find(name), name) for name in names if name in clean]
        if not matches:
            continue
        _, name = min(matches, key=lambda pair: (pair[0], -len(pair[1])))
        # 한 줄에 전후 값이 섞이면 자동으로 어느 열인지 추정하지 않는다.
        before, after = "정정 전" in clean, "정정 후" in clean
        if before and after:
            continue
        if rule["revision"] == "before" and not before:
            continue
        if rule["revision"] == "after" and before:
            continue
        if rule["period"] and rule["period"] not in clean:
            continue
        # 기간 여러 개를 가진 표·문장은 별도 의미 판정에 맡긴다.
        years = set(re.findall(r"20\d{2}", clean))
        if rule["period"] and "~" not in rule["period"] and len(years) > 1:
            continue
        candidates.append((line, clean[clean.find(name) + len(name):]))
    return candidates


def decimal_value(value) -> Decimal:
    """쉼표 없는 수치를 십진수로 바꿔 반올림 비교 오차를 피한다."""
    return Decimal(str(value).replace(",", ""))


def matches_value(observed: Decimal, expected, rule: dict) -> bool:
    """지정된 소수 자리와 허용 오차만 적용하여 수치를 비교한다."""
    expected = decimal_value(expected)
    if rule["rounding_digits"] is not None:
        quantum = Decimal(1).scaleb(-rule["rounding_digits"])
        observed = observed.quantize(quantum, rounding=ROUND_HALF_UP)
        expected = expected.quantize(quantum, rounding=ROUND_HALF_UP)
    tolerance = decimal_value(rule["tolerance"] or 0)
    return abs(observed - expected) <= tolerance


def arithmetic(expression: str) -> tuple[Decimal, Counter]:
    """숫자와 사칙연산만 허용하고 피연산자·연산자 구성도 반환한다."""
    if len(expression) > 300:
        raise ValueError("검사할 계산식이 너무 깁니다.")
    tree = ast.parse(expression, mode="eval")
    signature = Counter()

    def visit(node) -> Decimal:
        """안전한 AST 노드만 재귀 계산한다."""
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = decimal_value(node.value)
            signature[("operand", str(value.normalize()))] += 1
            return value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left), visit(node.right)
            signature[("operator", type(node.op).__name__)] += 1
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError("숫자와 사칙연산 외의 계산식은 검사하지 않습니다.")

    return visit(tree), signature


def calculation_result(rule: dict, candidates: list[tuple[str, str]]) -> tuple:
    """대상과 붙어 있는 명시적 등식의 피연산자·연산·결과·단위를 확인한다."""
    atom = rf"(?:{NUMBER})(?:\s*(?:{UNITS}))?"
    formula_pattern = re.compile(
        rf"(?P<formula>[\(\s]*{atom}(?:[\s\)\(]*[+\-*/×÷−][\s\(]*{atom})+[\s\)]*)"
        rf"\s*(?:=|≈|≒)\s*(?P<value>{NUMBER})\s*(?P<unit>{UNITS})"
    )
    found = [(line, match) for line, tail in candidates for match in formula_pattern.finditer(tail)]
    if len(found) != 1:
        return "undetermined", None, [], "대상에 연결된 계산식·결과·단위를 하나로 추출하지 못했습니다."
    line, match = found[0]
    expr = match["formula"].replace(",", "").replace("×", "*").replace("÷", "/").replace("−", "-")
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*(?:%|퍼센트)", lambda m: str(decimal_value(m[1]) / 100), expr)
    expr = re.sub(r"백만\s*원|원|주", "", expr).strip()
    observed = {"formula": expr, "value": float(decimal_value(match["value"])), "unit": normalized(match["unit"])}
    try:
        value, signature = arithmetic(expr)
    except ArithmeticError:
        return "fail", observed, [line], "대상에 연결된 식이 0으로 나누기 등으로 계산 불가능합니다."
    except (ValueError, SyntaxError):
        return "undetermined", observed, [line], "허용한 사칙연산 표현으로 계산식을 안정적으로 해석하지 못했습니다."
    _, expected_signature = arithmetic(rule["formula"])
    if signature != expected_signature:
        return "undetermined", observed, [line], "계산식의 피연산자·연산 형태가 골든 식과 달라 동치 여부를 자동 확정하지 않았습니다."
    passed = (
        normalized(match["unit"]) == normalized(rule["unit"] or "")
        and matches_value(value, rule["expected_value"], rule)
        and matches_value(decimal_value(match["value"]), value, rule)
    )
    return ("pass" if passed else "fail"), observed, [line], "명시된 피연산자·연산·산출값과 단위를 비교했습니다. 대상의 의미 해석은 Judge와 사람이 확인합니다."


def numeric_result(rule: dict, candidates: list[tuple[str, str]]) -> tuple:
    """명시적 대상 바로 뒤의 단일 수치 또는 범위와 단위를 추출한다."""
    connector = r"^[\s:|=\-]*(?:(?:은|는|이|가|으로|로)\s*)?"
    if rule["kind"] == "numeric_range":
        pattern = re.compile(connector + rf"(?P<min>{NUMBER})(?:\s*(?:{UNITS}))?\s*[~～–—-]\s*(?P<max>{NUMBER})\s*(?P<unit>{UNITS})")
    else:
        pattern = re.compile(connector + rf"(?P<value>{NUMBER})\s*(?P<unit>{UNITS})")
    found = [(line, match) for line, tail in candidates if (match := pattern.search(tail))]
    if len(found) != 1:
        return "undetermined", None, [], "대상·기간에 연결된 수치와 단위를 하나로 추출하지 못했습니다."
    line, match = found[0]
    # 같은 대상 후보에 다른 숫자가 존재하면 표나 복수 주장을 임의 해석하지 않는다.
    if len(candidates) != 1:
        return "undetermined", None, [line], "동일한 대상에 여러 후보가 있어 자동 확정하지 않았습니다."
    unit = normalized(match["unit"])
    if rule["kind"] == "numeric_range":
        observed = {"min": float(decimal_value(match["min"])), "max": float(decimal_value(match["max"])), "unit": unit}
        passed = all(matches_value(decimal_value(match[key]), rule["expected_value"][key], rule) for key in ("min", "max"))
    else:
        observed = {"value": float(decimal_value(match["value"])), "unit": unit}
        passed = matches_value(decimal_value(match["value"]), rule["expected_value"], rule)
    passed = passed and unit == normalized(rule["unit"] or "")
    return ("pass" if passed else "fail"), observed, [line], "대상 바로 뒤의 명시적 수치·단위를 비교했습니다. 의미·계획/실적 구분은 Judge와 사람이 확인합니다."


def evaluate_rules(question: GoldenQuestion, rag: dict) -> list[dict]:
    """계산·수치 추출이 불명확하면 undetermined로 남기고 인용 실존을 검사한다."""
    results = []
    generation = rag["generation"]
    for expected in question.rule_expectations:
        rule = expected.model_dump(mode="json")
        record = {
            "rule_id": rule["rule_id"], "evaluation_status": "skipped", "verdict": None,
            "expected": rule, "observed": None, "response_quotes": [],
            "reason": "정상 완료된 답변이 없어 규칙 검사를 실행하지 않았습니다.", "error": None,
        }
        if generation["execution_status"] != "completed" or not generation["response"]:
            results.append(record)
            continue
        try:
            response = generation["response"]
            if rule["kind"] == "citation_labels":
                used = re.findall(r"\[(E\d+)\]", response)
                known = {hit["label"] for hit in generation["contexts"]}
                missing = sorted(set(used) - known)
                verdict, observed, quotes, reason = (
                    "fail" if missing else "pass", {"used": used, "unknown": missing},
                    [f"[{label}]" for label in sorted(set(used))],
                    "실제 제공된 E 라벨의 존재만 확인했습니다. 주장 지지·인용 누락은 Judge와 사람이 확인합니다.",
                )
            elif rule["kind"] == "calculation":
                verdict, observed, quotes, reason = calculation_result(rule, target_lines(rule, response))
            else:
                verdict, observed, quotes, reason = numeric_result(rule, target_lines(rule, response))
            record.update(evaluation_status="completed", verdict=verdict, observed=observed,
                          response_quotes=quotes, reason=reason)
        except Exception as exc:
            record.update(evaluation_status="error", reason=str(exc),
                          error={"code": type(exc).__name__, "message": str(exc)})
        results.append(record)
    return results
