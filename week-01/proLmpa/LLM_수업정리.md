# LLM 수업 정리

## 1. LLM 개요

LLM(Large Language Model)은 입력 문맥 다음에 올 토큰의 확률을 계산하고, 토큰을 하나씩 이어 붙여 응답을 만드는 모델이다.

### 입력에서 응답까지

1. **토큰화** : 텍스트를 토큰 단위로 나누고 각 토큰을 정수 ID로 바꾼다.
2. **임베딩** : 토큰 ID를 의미를 담은 벡터로 변환한다.
3. **문맥 계산** : 앞선 토큰들과의 관계를 반영해 입력 문맥을 변환한다.
4. **확률 계산** : 어휘 전체에 대해 다음 토큰의 확률 분포를 구한다.
5. **토큰 선택** : temperature, top-p 같은 생성 설정에 따라 토큰 하나를 고른다.
6. **반복** : 고른 토큰을 문맥에 붙이고 3~5 단계를 반복한다. 종료 토큰, max_tokens, stop sequence 중 하나에 닿으면 멈춘다.
7. **디토큰화** : 토큰 열을 텍스트로 변환하여 응답한다.

핵심은 모델이 답을 찾는 것이 아니라 매 단계 확률적으로 다음 토큰을 고른다는 점이다. 그래서 같은 질문에도 답이 달라질 수 있고, 그럴 듯하지만 틀린 답(환각, hallucination)도 생긴다.

### API 호출에 필요한 요소

| 요소 | 역할 |
| --- | --- |
| 모델 | 사용할 모델 지정 |
| 메시지 (지시와 입력) | system : 역할과 규칙 / user : 요청 / assistant : 이전 응답 내역 |
| 생성 설정 | 응답 길이와 무작위성 제어 (ex. temperature, top-p, 출력 토큰 한도) |
| 출력 형식 | 응답 필드와 자료형 정의 |

```python
from vllm import LLM

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
conversation = [
    {
        "role": "system",
        "content": "You are a helpful assistant"
    },
    {
        "role": "user",
        "content": "Hello"
    },
    {
        "role": "assistant",
        "content": "Hello! How can I assist you today?"
    },
    {
        "role": "user",
        "content": "Write an essay about the importance of higher education.",
    },
]
outputs = llm.chat(conversation)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

정리하면 메시지는 프롬프트다. 문맥(context) 관리는 이 메시지에 무엇을 넣을지 정하는 것이고, core memory는 항상 넣는 핵심 정보, archival memory는 외부에 저장해 두고 필요할 때 검색해 넣는 정보다.

## 2. Pre-trained LLM 사용 시 기술적 한계

사전학습 모델은 학습 시점까지의 범용 데이터로, 정해진 길이의 입력 안에서, 확률적으로 다음 토큰을 고른다. 이 세가지 조건이 각각 아래 세 가지 한계로 이어진다.

### 한계 1. 도메인 지식과 최신 정보 부족 ("The LLM may underperform in your task.")

모델은 여러 분야를 두루 알지만 특정 분야를 깊이 알지 못하고, 학습 시점 이후 정보나 사내 문서는 아예 모른다. 고객사 전용 정책이나 지난달 출시된 기능을 물으면 모른다고 하기보다 그럴 듯한 오답(hallucination)을 내고, 출처도 제시하지 못한다.

### 한계 2. 컨텍스트 윈도우 제한 ("Context windows are limited.")

한 번에 처리할 수 있는 토큰 수가 정해져 있다. 수백 페이지 문서나 긴 대화 이력을 그대로 넣으면 한도를 넘거나, 들어가더라도 중간의 중요한 정보를 놓치고 비용과 지연이 커진다.

### 한계 3. 출력 제어의 어려움 ("LLMs are difficult to control.")

확률적 생성이기 때문에 형식, 톤, 안정성을 보장하지 어렵다. 실무에서는 JSON 형식이 깨지거나, 지시를 무시하거나, 프롬프트 인젝션에 휘둘리는 형태로 나타난다.

## 3. LLM 성능 개선 방안 : 모델(model) 및 맥락(context) 최적화

LLM의 성능은 1) 모델 자체를 바꾸거나, 2) 모델에 넣는 맥락(프롬프트, 외부 자료, 도구)을 개선하는 두 축으로 높일 수 있다. 그 중 맥락 최적화는 프롬프트 → RAG → AI agent → Multi agents 순으로 복잡도와 자율성이 커지며, 각각 앞서 기술한 LLM의 기술적 한계 중 하나 이상을 겨냥한다.

> 출처 : 12 Prompt Engineering Techniques - Cobus Greyling

### 프롬프트 엔지니어링 (Prompt engineering)

프롬프트는 LLM 모델에게 요청에 대한 자세한 맥락을 제공하는 기술이다. 비용이 가장 낮아 모든 개선의 출발점이 된다.

- 명확한 지시 작성
- 역할 부여 (acting)
- 작업을 여러 절차로 나눠 단계별로 해결 (Chain-of-Thought)
- 이전 프롬프트의 결과를 다음 프롬프트의 context로 사용 (Prompt chaining)
- 다른 사람의 프롬프트 참고하기 (Prompt templates)

**Zero-shot vs. Few-shot**

- **Zero-shot** : 어떤 맥락(context)도 제공하지 않고 원하는 결과를 요청하기
- **Few-shot** : 맥락(context)을 가르는 정확한 기준을 담은 예시를 제공하기

> 출처 : Fine-Tune LLMS: Between Full & Partial Fine Tuning - Medium

### 파인 튜닝 (Fine-tuning)

특정 작업 데이터로 가중치를 추가 학습하여 용어, 출력 형식, 판단 방식 같은 행동을 바꾼다. 반복적이고 정확도가 중요한 작업에 유리하다.

대신 라벨링 데이터와 학습 비용이 들고, 범용성이 떨어지며, 지식이 자주 바뀌면 재학습 부담이 크다. LoRA (Low-Rank Adaptation) 같은 부분 파인튜닝은 비용을 줄일 수 있다.

> 출처 : Retrieval Augmented Generation (RAG) - EDUCBA

### RAG (Retrieval-Augemented Generation)

지식 데이터를 모델 밖 문서에 두고 사용자 질의 시점에 찾아 넣는다. 문서를 청크 단위로 나눠 embedding 모델로 벡터화하고 벡터 DB에 저장한다(indexing). 질의와 유사한 청크를 찾아(Retrieval) 프롬프트에 붙이고(Augmentation) 답을 생성한다(Generation).

재학습 없이 문서만 갱신하면 최신 정보가 반영되고, 근거 출처를 함께 제시할 수 있다. 관련 chunk만 넣으면 되므로 컨텍스트도 아낀다. 지식은 RAG, 행동 양식은 파인 튜닝으로 나눠 생각하면 된다.

### Agent Workflows & MCP

> LLM 기반 애플리케이션이 여러 단계에 걸쳐 복잡한 업무를 처리하는 절차. -- Andrew Ng

LLM이 작업을 단계로 계획하고, 도구를 실행하고, 결과를 보고 다음 행동을 정하는 과정을 반복한다. 단계마다 필요한 맥락만 다루므로 컨텍스트 부담이 줄고, 메모리 관리가 이를 보조한다.

MCP는 도구와 데이터 소스를 LLM 애플리케이션에 연결하는 표준 프로토콜이다. 서비스마다 따로 만들던 연동을 하나의 규격으로 통일해 에이전트에 도구를 붙이는 비용을 줄인다.

### 멀티 에이전트

**Q.** 이미 workflows를 여러 절차로 나눴고, LLM을 여러 번 실행하고, 이미 tool까지 다 쥐어줬는데 왜 Multi-agent가 필요한 건가?
**A.** 병렬성

역할별 에이전트를 나눠 병렬(in parallel)로 실행한다. 각 에이전트가 자기 하위 작업의 컨텍스트만 유지하므로 단일 에이전트의 컨텍스트 한계를 넘는다. 대신 에이전트 간 조율 비용과 토큰 사용량이 늘어난다.

- AI agent 재사용성 (reusability)
- agent 아키텍처, 위계(hierarchy)
- 특정 구성요소별로 업데이트 가능
- 마이크로 단위 내결함성
- 최적화
- 문제 확인 쉬움 (Debugging)

## 생각해볼 거리

- 컨텍스트 윈도우가 100만 토큰 이상으로 커지면 RAG는 여전히 필요한가? 정확도와 비용 기준으로 언제 "전부 넣기"가 RAG보다 나은가?
- 단일 에이전트로 충분한 경우와 멀티 에이전트가 필요한 경우를 가르는 기준은 무엇인가? 병렬성 이득과 조율 비용은 어떻게 비교하는가?
- MCP와 API의 tools(function calling)는 어떤 관계인가? MCP가 tools를 대체하는지, 도구를 공급하는 방식만 표준화하는지 궁금하다.
- 파인튜닝과 RAG를 함께 쓸 때 역할을 어떻게 나누는가?
