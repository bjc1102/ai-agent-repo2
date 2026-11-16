# Week 03 — AWS GuardDuty RAG 평가 최종 제출

## 최종 상태와 평가 구분

최종 실행은 `results-live-03/evaluation_results.json`이며 생성 시각은 2026-11-14T05:39:29.122779+00:00이다. Q1~Q5 모두 실행 완료됐다. 과거 results-live-01/02의 연결 실패 기록은 이번 결과가 아니다.

사용자는 AWS GuardDuty 공식 문서와 생성 답변을 직접 검토하고 Q1~Q5 모두 적절하다고 확인했다. 이 확인에 따라 `results-live-03/human_review.json`에 reviewer `gkmin230`, 현재 기록 시각의 reviewed_at, verdict `success`, 문항별 이유와 근거 대응을 기록했다. reviewed_at은 사용자 지시에 따른 기록 시각이며 과거 실제 검토 시각을 추정한 값이 아니다. 에이전트가 독립적인 사람 검토를 했다고 주장하지 않는다.

자동 평가와 사람 판정은 별개다. **자동 평가 JSON은 바이트 단위로 보존**했고 재평가하거나 점수를 보정하지 않았다. 자동 목표 미달을 사람 성공 판정으로 덮어쓰지 않는다.

## 평가 구성과 성공 기준

원문은 [AWS GuardDuty EC2 finding types](https://docs.aws.amazon.com/guardduty/latest/ug/guardduty_finding-types-ec2.html)의 로컬 `data/guardduty.txt` 스냅샷이다. 근거 줄 번호는 이 스냅샷의 1-based inclusive 기준이며 현재 웹페이지 줄 번호와는 다를 수 있다. 제출본 `data/`에 원문과 기존 청크를 포함해 대조할 수 있게 했다.

| 설정 | 값 |
|---|---|
| 청킹 | 고정 길이 800자, overlap 100자, stride 700자; 73개 청크 |
| 임베딩 / 검색 | text-embedding-3-small / Chroma guardduty_docs, Top-3 |
| 생성 / Judge | gpt-4o-mini / gpt-4o-mini |
| temperature | 생성 0, Judge 0.01 |
| 평가 버전 | Ragas 0.3.5 |
| 지표 | LLMContextPrecisionWithReference, Faithfulness |
| DB 처리 | 원본 Chroma의 임시 복사본에서 검색 |

`questions.json`의 reference와 기대 답변은 생성 입력에 넣지 않는다. 생성에는 질문·검색 문서·일반 시스템 지시만 전달하고, reference는 생성 후 Judge에만 사용한다. 실제 messages, 검색 순위·거리·청크, 답변, Judge 출력과 usage는 자동 평가 JSON에 남아 있다.

사전에 정한 목표는 적용 문항의 Context Precision ≥ 0.8, Faithfulness ≥ 0.9, 숫자 환각 0건과 사람의 핵심 사실 통과다. Q1~Q3은 정의 설명, Q4는 근거 없는 회사 건수 거절, Q5는 정의 설명과 건수 거절의 분리를 확인한다. 5문항 결과를 운영 품질의 통계적 보장으로 해석하지 않는다.

## Q1~Q5 자동 점수와 사람 판정 비교

표의 점수는 가독성을 위해 소수 넷째 자리로 반올림했다. JSON의 전체 정밀도 값은 변경하지 않았다.

| 문항 | Context Precision | Faithfulness | 사람 판정 (gkmin230) | 핵심 확인 |
|---|---:|---:|---|---|
| Q1 SSHBruteForce 의미 | 1.0000 | 1.0000 | 성공 | SSH 무차별 대입, 피해 대상 Low / 공격 주체 High |
| Q2 Portscan 의미 | 1.0000 | 1.0000 | 성공 | outbound 스캔과 서비스/OS 식별; DoS 설명 혼입 없음 |
| Q3 RDPBruteForce 의미 | 0.8333 | 1.0000 | 성공 | RDP 무차별 대입, 역할별 심각도 구분 |
| Q4 어제 회사 탐지 건수 | N/A | N/A | 성공 | 문서만으로 확인 불가, 임의 수치 없음 |
| Q5 SSH 의미 + 회사 건수 | 0.0000 | 0.8333 | 성공 | SSH 설명 후 회사 건수 확인 불가를 구분 |

각 지표의 scored 분모는 4문항(Q1·Q2·Q3·Q5)이다. Context Precision 평균은 0.7083, Faithfulness 평균은 0.9583이다. 전체 지표 상태는 scored 8개, N/A 2개, 오류·미실행 0개다. Q5는 두 자동 지표 모두 개별 목표에 미달했고, Context Precision 평균도 목표 미달이다. 사람 판정은 5/5 성공이고 회사 건수를 만들어낸 답변은 없다.

Q4는 원문에 회사별 일별 탐지 기록이 없어 reference 기반 검색 관련성을 정의하지 않았다. Faithfulness도 이 거절 행동의 적절성 평가에서 사전 제외했다. N/A는 0점이나 성공 점수가 아니며 평균에서 제외한다. 거절과 숫자 환각은 사람 판정으로 확인한다.

## 사람 판정의 근거와 한계

- Q1: 원문 848~864행에 SSH 의미, 역할별 심각도와 포트 조건이 있다. chunk_066/067/069가 정의와 조치 근거를 제공한다. 답변은 포트 22 조건을 생략했지만 핵심 의미와 역할 구분을 충족한다. 피해 대상 설명은 뒤의 공격 주체 설명과 함께 읽는다.
- Q2: 569~578행이 방향·목적·심각도·데이터 소스의 근거이며 chunk_043/044에 검색됐다. 승인된 취약점 점검에 따른 오탐 가능성(580~583행)은 답변에서 생략됐지만 핵심 정의는 맞는다. 예상 밖 활동이면 손상 가능성이 있다는 답변은 원문 602~604행과 일치한다. 검색된 이전 유형/DoS의 유사 조치 문구를 Portscan 전용 근거라고 취급하지 않는다.
- Q3: 811~823행과 chunk_064가 RDP 의미 및 역할별 심각도를 뒷받침한다. Windows 명시는 생략됐지만 SSH와 혼동하지 않았다.
- Q4: 원문 전체의 범위는 탐지 유형 설명이며 회사 계정의 실제 일별 통계가 아니다. 답변은 확인 불가라고 거절했다. 필요한 계정 기록·집계 기간·시간대를 안내하면 더 좋지만 현재 거절은 적절하다. 부재를 특정 한 줄로 입증했다고 하지 않는다.
- Q5: 848~864행이 SSH 정의·포트 조건을 뒷받침한다. 회사 건수는 근거 부재를 명시했다. 원문 기준 적절성과 검색된 조각만으로 정의 전체를 입증할 수 있는지는 별도 문제다.

human_review의 source_support/retrieval_found_evidence/generation_used_evidence는 Q4에서 false다. 회사 건수 근거가 실제로 없기 때문이며 실패 판정이 아니다. answerability_handled는 전 문항 true다. Q5의 검색 근거 true는 부분적으로 유용한 SSH 근거를 찾았다는 뜻이며 완전 회수나 잡음 없음의 뜻이 아니다. 필드 의미와 사용자 확인 출처를 JSON에도 명시했다.

## Q2: DoS 잡음이 있는데 Context Precision이 1.0인 이유

이번 실행의 검색 순서는 chunk_043 → chunk_044 → chunk_010이다. Judge 관련성 verdict는 [1, 1, 0]이다. 043/044는 Portscan 정의와 설명을 포함하며 010은 TCP DoS와 다음 UDP DoS 도입부로 질문의 직접 근거가 아니다. Judge도 010을 무관하다고 판정했다.

[Ragas Context Precision](https://docs.ragas.io/en/v0.3.5/concepts/metrics/available_metrics/context_precision/)은 관련 청크가 얼마나 앞에 배치됐는지 평가한다. 관련 청크 위치의 precision@k를 관련 청크 수로 나누므로, [1,1,0]에서는 `(1×1 + 1×1 + (2/3)×0) / 2 = 1`이다. 저장된 실제 값은 분모 안정화 항으로 인해 `0.99999999995`다.

즉 1.0은 검색 결과 전체가 깨끗하다는 뜻이 아니다. 이 사례의 별도 단순 관련 청크 비율은 2/3, 잡음 비율은 1/3이다(설명용 집계이며 Ragas 원점수 대체가 아님). 답변에는 DoS나 대량 트래픽을 Portscan의 의미로 섞은 주장이 없고 Faithfulness는 1.0이다. 검색 잡음은 관측했으나 생성 실패라고 결론 내리지는 않는다. 잡음 유입 원인이 청킹 하나라고도 확정할 수 없다.

## Q5: 적절한 답변인데 Context Precision이 0.0인 이유

이번 검색 순서는 chunk_064(RDP 정의) → chunk_067(SSH 설명의 잘린 후반과 포트 조건) → chunk_002(탐지 유형 목록)이다. 핵심 SSH 정의 제목과 앞부분을 가진 chunk_066은 검색되지 않았다. 완전한 정의 근거가 부족하고 RDP·목록 잡음이 섞인 실제 검색 한계가 있다.

동시에 Judge의 관련성 verdict는 [0,0,0]이다. 두 번째 청크의 reason은 SSH 공격·모니터링 설명이 있다는 점을 인정하면서도 **어제 회사 탐지 건수를 답하지 못한다**는 이유로 무관 처리했다. `questions.json`은 SSH 의미만 Context Precision reference로 삼았지만 user_input에는 복합 질문 전체가 전달됐다. 기록된 reason으로 볼 때 Judge가 부분 질문 평가 범위를 일관되게 지키지 못한 것으로 해석된다. 0.0을 SSH 근거가 전혀 없다는 사실 판정으로 사용할 수 없다. 이 해석은 저장된 Judge 설명을 바탕으로 하며 내부 추론을 추정해 복원한 것이 아니다.

Faithfulness 원점수는 `0.8333333333333334`(6개 주장 중 5개 지지)다. Judge는 첫 SSH 정의 주장을 RDP 문맥 때문에 지지되지 않는다고 판단하고 나머지 포트·무단 접근·건수 거절 주장은 지지했다. 두 번째 Judge 출력은 코드 블록 안 JSON이라 단순 parsed_output이 null이지만 raw_output과 최종 점수는 남아 있다. 이를 평가 미실행으로 오인하지 않는다.

원문과 대조하고 사용자가 적절하다고 확인한 사람 판정은 성공이다. 이는 자동 지표를 무효화하거나 점수를 변경하는 행위가 아니다. 정의 문장이 잘려 검색된 실제 약점도 함께 남긴다. [Faithfulness](https://docs.ragas.io/en/v0.3.5/concepts/metrics/available_metrics/faithfulness/)는 검색 문맥 대비 주장 지지를 측정하므로 원문 전체 기준 정답성 및 거절의 적절성과 동일하지 않다.

후속 평가에서는 검색/생성에 원래 복합 질문을 유지한 채, SSH 부분의 Context Precision용 user_input과 reference를 같은 범위로 맞추고 건수 거절은 별도 기준으로 평가한다. 평가 입력 변경 결과는 새로운 run으로 저장해 기존 결과와 구분해야 한다. 이번에는 재평가하지 않았다.

## 고정 길이 청킹의 한계와 탐지 유형별 개선안

800자/100자 중복은 구현이 단순하지만 문서 구조와 문장 경계를 보장하지 않는다. chunk_067은 문장 중간에서 시작해 SSH 제목이 없고, chunk_043은 이전 유형의 조치와 Portscan 도입을 함께 담는다. chunk_069에는 SSH 조치와 다음 TorClient 도입이 섞인다. 100자 중복만으로 정의·심각도·역할·주의사항을 항상 함께 보존할 수 없다. 목차형 청크가 정의 청크처럼 검색되고 서로 비슷한 SSH/RDP/WinRM 표현이 혼동될 수 있다. 중복은 제한된 Top-3 자리를 소모할 수도 있다. 다만 현재 결과만으로 각각의 검색 순위 원인을 청킹으로 단정할 수 없다.

개선안은 아직 구현·재임베딩하지 않은 제안이다.

1. HTML의 탐지 유형 본문 제목/anchor를 경계로 `Recon:EC2/Portscan`, `UnauthorizedAccess:EC2/SSHBruteForce`, `UnauthorizedAccess:EC2/RDPBruteForce`별 부모 문서를 만든다. 목차는 제외하고 다음 유형 경계를 넘지 않는다. 텍스트 스냅샷을 처리할 경우 목차의 반복 이름을 본문 섹션으로 오인하지 않도록 설명·심각도·데이터 소스 패턴을 함께 확인한다.
2. 각 유형의 정의·Default severity·Data source·역할별 심각도 Note·포트 제약은 하나의 핵심 청크로 묶는다. 특히 SSH의 포트 22 조건과 RDP의 Windows 조건을 정의에서 분리하지 않는다.
3. 긴 Remediation은 문단/문장 경계의 자식 청크로 나눈다. 각 자식에 finding_type과 섹션 경로를 붙이고 역할별 조치 및 suppression 조건을 보존한다. 중복은 같은 유형 안에서 필요한 문장에만 적용하고 유형 간 중복은 만들지 않는다.
4. metadata에 finding_type, section, parent_id, source_url, 원문 줄/문자 범위, snapshot hash를 저장한다. 명시적인 탐지 유형명이 있는 질문은 이름 매칭과 벡터 검색을 결합하고, 관련 자식 검색 시 부모 핵심 정의를 확장해 전달한다. reranking과 중복 제거는 청킹 변경과 별도 실험으로 비교한다.
5. 별도 DB에서 같은 질문·모델·Top-3로 기존/구조 기반 청킹을 비교한다. 관련 청크 비율, 정의·심각도·포트 조건 근거 회수, 잡음 비율, 자동 점수·Judge 출력·사람 판정을 함께 본다. Q5는 평가 범위 정렬 실험을 별도 실행해 청킹 효과와 섞지 않는다. 원본 DB와 이번 자동 결과는 보존한다.

## 제출 파일 및 검증

제출본은 `submission/week-03/gkmin230/`에 같은 디렉터리 구조로 준비했다. 요청한 최종 목적지는 `~/ai-agent-repo2/week-03/gkmin230/`이지만 현재 실행 환경의 쓰기 권한 제한으로 해당 경로 생성이 거부됐다. 아래 파일만 복사하면 된다. 원본 저장소 다른 주차와 Git 메타데이터는 변경하지 않았다.

- `README.md`: 최신 비교표, 근거, 불일치 분석과 개선 제안
- `evaluate.py`, `requirements.txt`, `questions.json`: 기존 평가 코드와 질문/정답 기준
- `data/guardduty.txt`, `data/chunks.json`: 공개 문서 스냅샷과 기존 청킹 근거
- `results-live-03/evaluation_results.json`: 자동 평가 원본
- `results-live-03/human_review.json`: 사용자 확인에 따른 사람 판정
- `validation.json`: SHA-256 및 판정·집계·제출 제외 검증 결과

`.env`, API 키, 가상환경, Chroma DB, 캐시 및 과거 실패 결과는 제출본에 포함하지 않는다. DB·키가 없으므로 제출 폴더만으로 API 실행을 재현할 수는 없다. 재실행 시 원래 로컬 프로젝트의 data와 chroma_db를 `--project-root`로 지정하고 새 `--output`을 사용한다. 기존 결과를 재실행 대상으로 덮어쓰지 않는다. preflight.py는 원래 프로젝트 디렉터리 구조와 과거 결과 경로에 의존하므로 제출본에서 제외했다.

이번 마무리는 API 호출 없이 정적 검증했다. 자동 JSON SHA-256은 `7906e7f89a90228883357df13c93fb8bb18172fd117f20f3e53b488ebe4492f5`이며 사람 판정의 result_sha256과 일치한다. 문항·질문·답변·원문 해시·73개 청크 규칙·평균·N/A 분모·사람 판정 필수 필드를 검증했다. commit, push, PR은 수행하지 않았다.
