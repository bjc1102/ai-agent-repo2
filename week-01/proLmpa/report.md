# LLM 호출 실험 결과

## E1. effort low vs high

- 모델: claude-sonnet-5-5 / SDK: anthropic 2.3.0
- 바꾼 조건: output_config.effort (low / high)
- 유지한 조건: model=claude-sonnet-5-5, system, user(계산 문제), max_tokens=8000, 대화 이력 없음
- 실제 반복 횟수 (effort=low): 3회
- 실제 반복 횟수 (effort=high): 3회

| 조건 | 회차 | stop_reason | 입력/출력 토큰 | 지연(s) | 자동 점검 | 오류 | 응답 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| effort=low | 1 | end_turn | 122/348 | 3.23 | {"format_ok": true, "answer": 8003, "correct": true} | - | 1~200에서 3의 배수의 합, 5의 배수의 합, 15의 배수의 합을 구한다. ⏎  ⏎ 3의 배수: 3×1 ~ 3×66(198)이므로 합 = 3×(66×67/2) = 3×2211 = 6633 ⏎  ⏎ 5의 배수: 5×1 ~ 5×40(200)이므로 합 = 5×(40×41/2) = 5×82… |
| effort=low | 2 | end_turn | 122/374 | 3.22 | {"format_ok": true, "answer": 8003, "correct": true} | - | 1 이상 200 이하에서 3 또는 5의 배수의 합에서 15의 배수의 합을 두 번 빼면 된다. 3 또는 5로 나누어떨어지는 수 중 15의 배수는 3의 배수이면서 5의 배수이므로, 합집합에는 한 번 들어 있고 제외하려면 한 번만 빼면 된다. ⏎  ⏎ 구하는 값은 (3의 배수의 합) + (5… |
| effort=low | 3 | end_turn | 122/355 | 3.1 | {"format_ok": true, "answer": 8003, "correct": true} | - | 1~200에서 3의 배수의 합, 5의 배수의 합, 15의 배수의 합을 구한다. 구하는 값은 (3의 배수 합) + (5의 배수 합) − 2×(15의 배수 합)이다. 3 또는 5의 배수 합은 두 합에서 15의 배수 합을 한 번 빼서 구하고, 15의 배수는 다시 제외해야 하므로 한 번 더 빼… |
| effort=high | 1 | end_turn | 122/411 | 3.43 | {"format_ok": true, "answer": 8003, "correct": true} | - | 1부터 200까지에서 3의 배수의 합, 5의 배수의 합, 15의 배수의 합을 구한다. 3 또는 5의 배수이면서 15의 배수가 아닌 수는 (3의 배수 합) + (5의 배수 합) − 2×(15의 배수 합)으로 구할 수 있다. ⏎  ⏎ 3의 배수의 합 계산: ⏎ - 3의 배수는 3×1부터 3… |
| effort=high | 2 | end_turn | 122/465 | 3.29 | {"format_ok": true, "answer": 8003, "correct": true} | - | 1부터 200까지에서 3의 배수의 합, 5의 배수의 합, 15의 배수의 합을 각각 구한다. ⏎  ⏎ - 3의 배수: 3×1 ~ 3×66(=198)이므로 합은 3 × (66×67/2) = 3 × 2211 = 6633 ⏎ - 5의 배수: 5×1 ~ 5×40(=200)이므로 합은 5 × (4… |
| effort=high | 3 | end_turn | 122/422 | 3.4 | {"format_ok": true, "answer": 8003, "correct": true} | - | 1부터 200까지에서 3의 배수의 합, 5의 배수의 합, 15의 배수의 합을 구한다. 3 또는 5의 배수이면서 15의 배수가 아닌 수의 합은 (3의 배수의 합) + (5의 배수의 합) − 2×(15의 배수의 합)이다. ⏎  ⏎ 3 또는 5로 나누어떨어지는 수의 합은 (3의 배수의 합) … |

**관찰 (직접 작성, 3~5줄)**

- 

## E2. zero-shot vs few-shot

- 모델: claude-sonnet-5-5 / SDK: anthropic 2.3.0
- 바꾼 조건: 예시 3개 포함 여부
- 유지한 조건: model=claude-sonnet-5-5, system, 지시문, 분류 대상 리뷰 5개, effort=high, max_tokens=8000
- 실제 반복 횟수 (zero-shot): 3회
- 실제 반복 횟수 (few-shot): 3회

| 조건 | 회차 | stop_reason | 입력/출력 토큰 | 지연(s) | 자동 점검 | 오류 | 응답 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero-shot | 1 | end_turn | 222/179 | 2.44 | {"format_ok": true, "labels": "부정/긍정/중립/부정/긍정"} | - | 부정 ⏎ 긍정 ⏎ 중립 ⏎ 부정 ⏎ 긍정 |
| zero-shot | 2 | end_turn | 222/216 | 2.81 | {"format_ok": true, "labels": "중립/긍정/중립/부정/긍정"} | - | 중립 ⏎ 긍정 ⏎ 중립 ⏎ 부정 ⏎ 긍정 |
| zero-shot | 3 | end_turn | 222/220 | 2.71 | {"format_ok": true, "labels": "부정/긍정/중립/부정/긍정"} | - | 부정 ⏎ 긍정 ⏎ 중립 ⏎ 부정 ⏎ 긍정 |
| few-shot | 1 | end_turn | 328/192 | 2.51 | {"format_ok": true, "labels": "부정/긍정/중립/부정/긍정"} | - | 부정 ⏎ 긍정 ⏎ 중립 ⏎ 부정 ⏎ 긍정 |
| few-shot | 2 | end_turn | 328/207 | 2.41 | {"format_ok": true, "labels": "부정/긍정/중립/부정/긍정"} | - | 부정 ⏎ 긍정 ⏎ 중립 ⏎ 부정 ⏎ 긍정 |
| few-shot | 3 | end_turn | 328/236 | 2.87 | {"format_ok": true, "labels": "부정/긍정/중립/부정/긍정"} | - | 부정 ⏎ 긍정 ⏎ 중립 ⏎ 부정 ⏎ 긍정 |

- zero-shot: 서로 다른 라벨 조합 2종 → 부정/긍정/중립/부정/긍정, 중립/긍정/중립/부정/긍정
- few-shot: 서로 다른 라벨 조합 1종 → 부정/긍정/중립/부정/긍정

**관찰 (직접 작성, 3~5줄)**

- 

## E3. 프롬프트 JSON 지시 vs 구조화 출력

- 모델: claude-sonnet-5-5 / SDK: anthropic 2.3.0
- 바꾼 조건: output_config.format(JSON 스키마) 지정 여부
- 유지한 조건: model=claude-sonnet-5-5, system, user(같은 지시문과 문단), effort=high, max_tokens=8000
- 실제 반복 횟수 (prompt-only): 3회
- 실제 반복 횟수 (json_schema): 3회

| 조건 | 회차 | stop_reason | 입력/출력 토큰 | 지연(s) | 자동 점검 | 오류 | 응답 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prompt-only | 1 | end_turn | 228/52 | 1.58 | {"json_ok": true, "keys_ok": true, "budget_ok": true} | - | {"date": "2026-03-12", "organization": "디지털정책과", "budget_krw": 450000000, "deadline": "2026-06-말"} |
| prompt-only | 2 | end_turn | 228/52 | 1.42 | {"json_ok": true, "keys_ok": true, "budget_ok": true} | - | {"date": "2026-03-12", "organization": "디지털정책과", "budget_krw": 450000000, "deadline": "2026-06-30"} |
| prompt-only | 3 | end_turn | 228/54 | 1.42 | {"json_ok": true, "keys_ok": true, "budget_ok": true} | - | {"date": "2026-03-12", "organization": "디지털정책과", "budget_krw": 450000000, "deadline": "2026년 6월 말"} |
| json_schema | 1 | end_turn | 493/51 | 2.93 | {"json_ok": true, "keys_ok": true, "budget_ok": true} | - | {"date":"2026-03-12","organization":"디지털정책과","budget_krw":450000000,"deadline":"2026년 6월 말"} |
| json_schema | 2 | end_turn | 493/49 | 1.23 | {"json_ok": true, "keys_ok": true, "budget_ok": true} | - | {"date":"2026-03-12","organization":"디지털정책과","budget_krw":450000000,"deadline":"2026-06-30"} |
| json_schema | 3 | end_turn | 493/49 | 1.02 | {"json_ok": true, "keys_ok": true, "budget_ok": true} | - | {"date":"2026-03-12","organization":"디지털정책과","budget_krw":450000000,"deadline":"2026-06-30"} |

**관찰 (직접 작성, 3~5줄)**

- 

## E4. temperature 지정 시 동작 확인

- 모델: claude-sonnet-5-5 / SDK: anthropic 2.3.0
- 바꾼 조건: temperature 파라미터 지정 여부 (미지정 / 0.7)
- 유지한 조건: model=claude-sonnet-5-5, system, user, effort=high, max_tokens=8000
- 실제 반복 횟수 (no-temperature): 6회
- 실제 반복 횟수 (temperature=0.7): 3회

| 조건 | 회차 | stop_reason | 입력/출력 토큰 | 지연(s) | 자동 점검 | 오류 | 응답 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no-temperature | 1 | end_turn | 44/86 | 1.92 | {} | - | LLM(대규모 언어 모델)은 방대한 양의 텍스트 데이터로 학습하여 사람의 언어를 이해하고 문장을 생성하거나 질문에 답할 수 있는 인공지능 모델입니다. |
| no-temperature | 2 | end_turn | 44/84 | 1.79 | {} | - | LLM(대규모 언어 모델)은 방대한 텍스트 데이터를 학습해 문맥을 이해하고 사람처럼 글을 이해·생성할 수 있도록 만들어진 인공지능 모델입니다. |
| no-temperature | 3 | end_turn | 44/110 | 1.76 | {} | - | LLM(대규모 언어 모델)은 방대한 양의 텍스트 데이터를 학습해 사람의 언어를 이해하고 문장을 생성하며 질문에 답하거나 번역·요약 같은 다양한 언어 작업을 수행할 수 있는 인공지능 모델입니다. |
| no-temperature | 1 | end_turn | 44/88 | 2.02 | {} | - | LLM(대규모 언어 모델)은 방대한 양의 텍스트 데이터로 학습하여 문맥을 이해하고 사람처럼 글을 생성·요약·번역·답변할 수 있는 인공지능 모델입니다. |
| no-temperature | 2 | end_turn | 44/100 | 1.7 | {} | - | LLM(대규모 언어 모델)은 방대한 텍스트 데이터를 학습하여 사람의 언어를 이해하고 문장을 생성하며 질문에 답하는 등 다양한 언어 작업을 수행할 수 있는 인공지능 모델입니다. |
| no-temperature | 3 | end_turn | 44/77 | 1.6 | {} | - | LLM(대규모 언어 모델)은 방대한 양의 텍스트 데이터로 학습하여 사람의 언어를 이해하고 생성할 수 있도록 만든 인공지능 모델입니다. |
| temperature=0.7 | 1 | - | - | 0.0 | - | None: TypeError: Messages.create() got an unexpected keyword argument 'temperature' | - |
| temperature=0.7 | 2 | - | - | 0.0 | - | None: TypeError: Messages.create() got an unexpected keyword argument 'temperature' | - |
| temperature=0.7 | 3 | - | - | 0.0 | - | None: TypeError: Messages.create() got an unexpected keyword argument 'temperature' | - |

**관찰 (직접 작성, 3~5줄)**

- 

## E5. effort 전 단계 비교 (low/medium/high/xhigh/max)

- 모델: claude-sonnet-5-5 / SDK: anthropic 2.3.0
- 바꾼 조건: output_config.effort (low/medium/high/xhigh/max)
- 유지한 조건: model=claude-sonnet-5-5, system, user(소수 합산), max_tokens=8000
- 실제 반복 횟수 (effort=low): 5회
- 실제 반복 횟수 (effort=medium): 5회
- 실제 반복 횟수 (effort=high): 5회
- 실제 반복 횟수 (effort=xhigh): 5회
- 실제 반복 횟수 (effort=max): 5회

| 조건 | 회차 | stop_reason | 입력/출력 토큰 | 지연(s) | 자동 점검 | 오류 | 응답 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| effort=low | 1 | end_turn | 104/239 | 2.52 | {"format_ok": true, "answer": 3167, "correct": true} | - | 100 이상 200 이하의 소수는 다음과 같다. ⏎  ⏎ 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199 ⏎  ⏎ (100과 200은 합성수이므로 제… |
| effort=low | 2 | end_turn | 104/247 | 2.54 | {"format_ok": true, "answer": 3167, "correct": true} | - | 100 이상 200 이하의 소수는 다음과 같다. ⏎  ⏎ 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199 ⏎  ⏎ (100과 200은 합성수이므로 제… |
| effort=low | 3 | end_turn | 104/259 | 2.46 | {"format_ok": true, "answer": 3167, "correct": true} | - | 100 이상 200 이하의 소수는 다음과 같다. ⏎  ⏎ 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199 ⏎  ⏎ (100과 200은 합성수이므로 제… |
| effort=low | 4 | end_turn | 104/440 | 3.21 | {"format_ok": true, "answer": 3167, "correct": true} | - | 100 이상 200 이하의 소수는 다음과 같다. ⏎  ⏎ 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199 ⏎  ⏎ 이 소수들을 차례로 더한다. ⏎  … |
| effort=low | 5 | end_turn | 104/283 | 2.96 | {"format_ok": true, "answer": 3167, "correct": true} | - | 100 이상 200 이하의 소수는 다음과 같다. ⏎  ⏎ 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199 ⏎  ⏎ (100과 200은 합성수이므로 제… |
| effort=medium | 1 | end_turn | 104/343 | 2.74 | {"format_ok": true, "answer": 3167, "correct": true} | - | 100 이상 200 이하의 소수는 다음과 같다. ⏎  ⏎ 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199 ⏎  ⏎ (100과 200은 소수가 아니다.… |
| effort=medium | 2 | end_turn | 104/265 | 2.57 | {"format_ok": true, "answer": 3167, "correct": true} | - | 100 이상 200 이하의 소수는 다음과 같다. ⏎  ⏎ 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199 ⏎  ⏎ (100과 200은 합성수이므로 제… |
| effort=medium | 3 | end_turn | 104/274 | 2.73 | {"format_ok": true, "answer": 3167, "correct": true} | - | 100 이상 200 이하의 소수는 다음과 같다. ⏎  ⏎ 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199 ⏎  ⏎ 합을 구간별로 나누어 계산한다. ⏎… |
| effort=medium | 4 | end_turn | 104/269 | 2.58 | {"format_ok": true, "answer": 3167, "correct": true} | - | 100 이상 200 이하의 소수는 다음과 같다. ⏎  ⏎ 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199 ⏎  ⏎ (100과 200은 합성수이므로 제… |
| effort=medium | 5 | end_turn | 104/274 | 2.66 | {"format_ok": true, "answer": 3167, "correct": true} | - | 100 이상 200 이하의 소수는 다음과 같다. ⏎  ⏎ 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199 ⏎  ⏎ 합을 구하면 다음과 같다. ⏎  ⏎… |
| effort=high | 1 | end_turn | 104/438 | 3.06 | {"format_ok": true, "answer": 3167, "correct": true} | - | 100 이상 200 이하의 소수는 다음 21개다. ⏎  ⏎ 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199 ⏎  ⏎ 차례로 더하면 다음과 같다. ⏎ … |
| effort=high | 2 | end_turn | 104/525 | 3.53 | {"format_ok": true, "answer": 3167, "correct": true} | - | 100 이상 200 이하의 소수는 다음과 같다. ⏎  ⏎ 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199 (총 21개) ⏎  ⏎ 이들을 차례로 더하면… |
| effort=high | 3 | end_turn | 104/610 | 3.65 | {"format_ok": true, "answer": 3167, "correct": true} | - | 100 이상 200 이하의 소수는 다음 21개입니다. ⏎  ⏎ 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199 ⏎  ⏎ (200 이하의 소수는 46개… |
| effort=high | 4 | end_turn | 104/518 | 3.66 | {"format_ok": true, "answer": 3167, "correct": true} | - | 100 이상 200 이하의 소수는 다음 21개다. ⏎  ⏎ 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199 ⏎  ⏎ (200 이하의 소수는 46개이고… |
| effort=high | 5 | end_turn | 104/523 | 3.41 | {"format_ok": true, "answer": 3167, "correct": true} | - | 100 이상 200 이하의 소수는 다음 21개다. ⏎  ⏎ 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199 ⏎  ⏎ 이들을 차례로 더하면 다음과 같다… |
| effort=xhigh | 1 | end_turn | 104/576 | 3.92 | {"format_ok": true, "answer": 3167, "correct": true} | - | 100 이상 200 이하의 소수는 다음 21개다. ⏎  ⏎ 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199 ⏎  ⏎ 이 수들을 차례로 더하면 다음과 … |
| effort=xhigh | 2 | end_turn | 104/587 | 4.01 | {"format_ok": true, "answer": 3167, "correct": true} | - | 100 이상 200 이하의 소수는 다음과 같다. ⏎  ⏎ 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199 (총 21개) ⏎  ⏎ 이 수들을 차례로 더… |
| effort=xhigh | 3 | end_turn | 104/569 | 3.78 | {"format_ok": true, "answer": 3167, "correct": true} | - | 100 이상 200 이하의 소수는 다음과 같다. ⏎  ⏎ 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199 (모두 21개) ⏎  ⏎ 이 수들을 차례로 … |
| effort=xhigh | 4 | end_turn | 104/637 | 4.39 | {"format_ok": true, "answer": 3167, "correct": true} | - | 100 이상 200 이하의 소수는 √200 ≈ 14.1 이하의 소수 2, 3, 5, 7, 11, 13으로 나누어떨어지지 않는 수를 골라 찾는다. ⏎  ⏎ **소수 목록 (21개):** ⏎ 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, … |
| effort=xhigh | 5 | end_turn | 104/638 | 4.23 | {"format_ok": true, "answer": 3167, "correct": true} | - | **100 이상 200 이하의 소수** ⏎  ⏎ 200 이하의 소수 판별에는 √200 ≈ 14.1 이하의 소수 2, 3, 5, 7, 11, 13으로 나누어 보면 충분하다. 이 소수들로 나누어떨어지지 않는 수를 찾으면 다음 21개다. ⏎  ⏎ 101, 103, 107, 109, 113, … |
| effort=max | 1 | end_turn | 104/1898 | 11.01 | {"format_ok": true, "answer": 3167, "correct": true} | - | 100 이상 200 이하의 소수를 찾으려면 √200 ≈ 14.1 이므로 2, 3, 5, 7, 11, 13으로 나누어떨어지는지만 확인하면 된다. ⏎  ⏎ **소수 목록 (21개)** ⏎  ⏎ 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,… |
| effort=max | 2 | end_turn | 104/2649 | 15.15 | {"format_ok": true, "answer": 3167, "correct": true} | - | **100 이상 200 이하의 소수** ⏎  ⏎ 200의 제곱근이 약 14.1이므로 2, 3, 5, 7, 11, 13으로 나누어 떨어지는지 확인하면 된다. 짝수와 5의 배수를 먼저 제외하고, 남은 수를 나눗셈으로 검사하면 소수는 다음 21개이다. ⏎  ⏎ 101, 103, 107, 10… |
| effort=max | 3 | end_turn | 104/1522 | 8.98 | {"format_ok": true, "answer": 3167, "correct": true} | - | ## 풀이 ⏎  ⏎ 200의 제곱근은 약 14.1이므로, 100~200 사이의 수가 소수인지는 2, 3, 5, 7, 11, 13으로 나누어 보면 판별할 수 있다. ⏎  ⏎ 홀수 중 합성수를 제외하면 소수는 다음 21개이다. ⏎  ⏎ 101, 103, 107, 109, 113, 127, … |
| effort=max | 4 | end_turn | 104/1456 | 8.52 | {"format_ok": true, "answer": 3167, "correct": true} | - | **100 이상 200 이하의 소수** ⏎  ⏎ 각 수가 2, 3, 5, 7, 11, 13으로 나누어떨어지는지 확인하면 된다. 14²=196이므로 13까지만 확인해도 충분하다. ⏎  ⏎ 구간별 소수는 다음과 같다. ⏎  ⏎ \| 구간 \| 소수 \| 구간 합 \| ⏎ \|---\|---… |
| effort=max | 5 | end_turn | 104/1411 | 8.23 | {"format_ok": true, "answer": 3167, "correct": true} | - | **풀이** ⏎  ⏎ 200의 제곱근은 약 14.1이므로, 100~200의 각 수가 2, 3, 5, 7, 11, 13으로 나누어떨어지는지 확인하면 소수 여부를 판별할 수 있다. ⏎  ⏎ 십의 자리 구간별 소수는 다음과 같다. ⏎  ⏎ \| 구간 \| 소수 \| 합 \| ⏎ \|---\|… |

**관찰**

- effort=low·medium 구간에서는 thinking_tokens=0으로 내부 추론 없이 응답하며, effort=high부터 thinking이 처음 활성화된다(평균 214 토큰). effort=max에서는 평균 1,337 thinking 토큰·10.38s로, low 대비 출력 토큰 약 6배·지연 약 3.8배가 소비됐다.
- 정확도는 5단계 모두 5/5(100%), 정답 3167로 일치했다. 소수 판별·합산이라는 검산 불가 태스크에서도 effort 수준에 관계없이 오답이 발생하지 않았다.
- 비용·속도 최적화 관점에서는 effort=low 또는 medium이 최선이다. thinking 임계점(high) 이상은 출력이 길어지고 중간 계산 과정을 상세히 서술하나 최종 정답은 동일하여 이 태스크에서 추가 effort는 효과가 없었다.
- effort=max의 응답은 구간별 소수 표 등 시각적 구조를 활용해 서술 방식의 가독성이 높아졌으나, 그것이 정확도 차이가 아님을 수치로 확인했다.

## E6. stop_sequences 유무 비교

- 모델: claude-sonnet-5-5 / SDK: anthropic 2.3.0
- 바꾼 조건: stop_sequences (없음 / ['2.'])
- 유지한 조건: model=claude-sonnet-5-5, system, user(번호 목록 3개), max_tokens=8000, output_config 없음
- 실제 반복 횟수 (no-stop): 3회
- 실제 반복 횟수 (stop-at-2): 3회

| 조건 | 회차 | stop_reason | 입력/출력 토큰 | 지연(s) | 자동 점검 | 응답 |
| --- | --- | --- | --- | --- | --- | --- |
| no-stop | 1 | end_turn | 57/264 | 2.33 | {"item_count": 3} | 1. **대규모 데이터 학습**: LLM은 방대한 양의 텍스트 데이터를 사전 학습하여 언어의 문법, 의미, 맥락과 폭넓은 지식을 습득합니다. ⏎ 2. **트랜스포머 기반 구조**: 대부분의 LLM은 어텐션 메커니즘을 활용하는 트랜스포머 아키텍처를 사용해 문장 내 단어 간의 관계와 긴 문… |
| no-stop | 2 | end_turn | 57/283 | 2.49 | {"item_count": 3} | 1. **대규모 데이터 학습**: LLM은 방대한 양의 텍스트 데이터를 사전 학습하여 언어의 문법, 의미, 맥락, 그리고 다양한 분야의 지식을 폭넓게 습득합니다. ⏎  ⏎ 2. **트랜스포머 기반 구조**: 대부분의 LLM은 어텐션 메커니즘을 활용하는 트랜스포머 아키텍처를 바탕으로 하여… |
| no-stop | 3 | end_turn | 57/257 | 2.34 | {"item_count": 3} | 1. **대규모 데이터 학습**: LLM은 방대한 양의 텍스트 데이터를 학습하여 언어의 문법, 의미, 맥락과 지식을 폭넓게 익힙니다. ⏎ 2. **트랜스포머 기반 구조**: 대부분의 LLM은 트랜스포머 아키텍처의 어텐션 메커니즘을 활용해 문장 내 단어 간의 관계와 긴 문맥을 효과적으로 … |
| stop-at-2 | 1 | stop_sequence | 57/174 | 1.80 | {"item_count": 1} | 1. **대규모 데이터 학습**: LLM은 방대한 양의 텍스트 데이터를 학습하여 언어의 문법, 의미, 맥락과 세상에 대한 폭넓은 지식을 습득합니다. ⏎  ⏎  |
| stop-at-2 | 2 | stop_sequence | 57/138 | 1.49 | {"item_count": 1} | 1. **대규모 데이터 학습**: LLM은 방대한 양의 텍스트 데이터를 학습하여 언어의 문법, 의미, 맥락과 지식을 폭넓게 익힙니다. ⏎  |
| stop-at-2 | 3 | stop_sequence | 57/149 | 1.41 | {"item_count": 1} | 1. **대규모 데이터 학습**: LLM은 방대한 양의 텍스트 데이터를 사전 학습하여 언어의 문법, 지식, 맥락적 패턴을 폭넓게 이해합니다. ⏎  ⏎  |

**관찰**

- stop_sequences=["2."] 지정 시 3회 모두 1번 항목 직후에서 정확히 절단되었고, stop_reason이 `stop_sequence`로 기록되어 `end_turn`과 프로그래밍적으로 명확히 구분된다.
- 출력 토큰은 평균 268 → 154로 약 43% 감소, 지연은 2.39s → 1.57s로 약 34% 단축됐다. 모델이 토큰을 생성하다 중단되므로 절감 효과는 stop 지점이 앞쪽일수록 비례해 커진다.
- stop_sequences는 응답 길이 제어와 비용 절감 수단으로 유효하며, 특히 스트리밍 파이프라인에서 원하는 지점에서 처리를 중단하는 용도로 적합하다.

## E7. thinking.type 비교 (disabled/adaptive/enabled)

- 모델: claude-sonnet-5-5 / SDK: anthropic 2.3.0
- 바꾼 조건: thinking.type (disabled / adaptive / enabled)
- 유지한 조건: model=claude-sonnet-5-5, system, user(감성 분류), effort=high, max_tokens=8000
- 실제 반복 횟수 (thinking=adaptive): 3회
- thinking=disabled / thinking=enabled: API 400으로 전량 실패 (아래 비고 참조)

| 조건 | 회차 | stop_reason | 입력/출력 토큰 | 지연(s) | 자동 점검 | 응답 |
| --- | --- | --- | --- | --- | --- | --- |
| thinking=adaptive | 1 | end_turn | 328/234 | 3.10 | {"format_ok": true, "labels": "부정/긍정/중립/부정/긍정"} | 부정 ⏎ 긍정 ⏎ 중립 ⏎ 부정 ⏎ 긍정 |
| thinking=adaptive | 2 | end_turn | 328/178 | 2.16 | {"format_ok": true, "labels": "부정/긍정/중립/부정/긍정"} | 부정 ⏎ 긍정 ⏎ 중립 ⏎ 부정 ⏎ 긍정 |
| thinking=adaptive | 3 | end_turn | 328/195 | 2.52 | {"format_ok": true, "labels": "부정/긍정/중립/부정/긍정"} | 부정 ⏎ 긍정 ⏎ 중립 ⏎ 부정 ⏎ 긍정 |

- thinking=adaptive: thinking_tokens 평균 184.3 | 라벨 조합 1종

**API 오류 비고**
- `thinking=disabled` → 400: "To turn thinking off on this model, send `thinking: {type: between_tools}` instead of `{type: disabled}`"
- `thinking=enabled` → 400: "`thinking.type.enabled` is not supported for this model. Use `thinking.type.adaptive` and `output_config.effort` to control thinking behavior."

**관찰**

- claude-sonnet-5-5에서 유효한 thinking.type 값은 `adaptive`뿐이다. `disabled` 대신 `between_tools`를 사용하라는 API 안내가 반환되었고, `enabled`는 이 모델에서 지원하지 않으며 thinking 강도는 `output_config.effort`로 제어해야 한다.
- thinking=adaptive 3회 모두 라벨 조합이 부정/긍정/중립/부정/긍정으로 동일하여 E2 few-shot과 같은 수준의 일관성을 보였다. thinking_tokens 평균 184.3으로 내부 추론이 이뤄졌으며, 이 값은 output_tokens에 포함되어 집계된다.
- thinking 강도를 올리려면 thinking.type이 아닌 output_config.effort를 높여야 한다는 것이 E5 결과와 일치한다. thinking.type은 추론 허용 모드(on·off·도구 사이)를 지정하는 스위치이고, 실제 추론 깊이는 effort가 결정한다.
