---
layout: '../../layouts/Post.astro'
title: 'Mistral API와 Vibe로 배우는 AI 활용 Cookbook'
description: 'Mistral API 공식 SDK로 채팅·스트리밍·구조화 출력·도구 호출·RAG·OCR까지직접 해 보고, 코딩 에이전트 Vibe로 확장하는 방법을 레시피별로 정리했습니다.'
date: '2026-11-10'
tags: ['mistral', 'api', 'llm', 'vibe', 'agents']
---

Mistral API를 "읽기만 하지 않고 직접 손에 묻히는" 가장 빠른 길은 예제를 따라 치는 것입니다. 이 글은 공식 Python SDK(`mistralai`) 기준으로, 채팅부터 RAG까지 레시피 6개와 코딩 에이전트 Vibe 활용 팁까지まとめました. API 키는 코드에 넣지 말고 환경변수로 관리하세요.

## 0. 준비하기

API 키는 [La Plateforme 콘솔](https://console.mistral.ai/)에서 발급받습니다 (Workspace → API keys). 터미널에서 환경변수를 설정하고 SDK를 설치합니다.

```bash
export MISTRAL_API_KEY="your-api-key"
pip install mistralai
```

```python
import os
from mistralai import Mistral

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
```

## 1. 첫 Chat Completion

가장 작은 호출입니다. `mistral-small-latest`는 빠르고 저렴하고, `mistral-large-latest`는 복잡한 추론에 강합니다.

```python
response = client.chat.complete(
    model="mistral-small-latest",
    messages=[{"role": "user", "content": "한 문장으로 LLM을 설명해 줘."}],
)
print(response.choices[0].message.content)
```

시스템 프롬프트로 말투와 출력 형식을 고정할 수 있습니다.

```python
messages = [
    {"role": "system", "content": "너는 한국어로 답하는 간결한 조수다. 코드는 코드 블록으로 감싼다."},
    {"role": "user", "content": "RAG가 뭐야?"},
]
response = client.chat.complete(model="mistral-large-latest", messages=messages)
```

## 2. 스트리밍으로 토큰 단위 출력하기

긴 답변을 실시간으로 보여줄 때 씁니다. 챗봇 UI의 기본 패턴이에요.

```python
stream = client.chat.stream(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": "제주도 3박 4일 여행 일정을 짜 줘."}],
)
for chunk in stream:
    delta = chunk.data.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
```

## 3. 구조화 출력 (JSON 강제)

`response_format`에 JSON Schema를 주면 모델이 스키마를 따르는 JSON만 반환합니다. 다음 파이프라인 단계로 바로 넘기기 좋습니다.

```python
import json

schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["title", "tags", "summary"],
}

response = client.chat.complete(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": "이 글을 JSON으로 정리해 줘: 오늘 서울 날씨는 맑고 미세먼지가 보통이다."}],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "article_summary", "schema": schema, "strict": True},
    },
)
data = json.loads(response.choices[0].message.content)
print(data["title"], data["tags"])
```

## 4. 도구 호출 (Function Calling)

모델이 외부 함수 호출을 "요청"하면 내 코드가 실제로 실행하고, 결과를 다시 모델에 넣어 최종 답을 만듭니다.

```python
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "특정 도시의 현재 날씨를 조회한다",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "도시 이름 (예: 서울)"}},
                "required": ["city"],
            },
        },
    }
]


def get_weather(city: str) -> str:
    table = {"서울": "맑음, 12도", "부산": "흐림, 15도", "제주": "비, 17도"}
    return table.get(city, "날씨 정보 없음")


messages = [{"role": "user", "content": "서울이랑 부산 날씨 비교해 줘."}]
response = client.chat.complete(model="mistral-large-latest", messages=messages, tools=tools, tool_choice="auto")
message = response.choices[0].message

for call in message.tool_calls or []:
    args = json.loads(call.function.arguments)
    result = get_weather(**args)
    messages.append(message.model_dump())
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.function.name,
            "content": result,
        }
    )

final = client.chat.complete(model="mistral-large-latest", messages=messages, tools=tools)
print(final.choices[0].message.content)
```

## 5. 임베딩으로簡易 RAG 만들기

문서를 벡터로 바꿔 두고, 질문과 가장 가까운 청크를 컨텍스트로 넣어 답하게 합니다. 임베딩 모델은 `mistral-embed`입니다.

```python
import math

corpus = [
    "Mistral API는 La Plateforme 콘솔에서 키를 발급받아 사용한다.",
    "mistral-small-latest는 빠르고 저렴해서 프로토타이핑에 적합하다.",
    "스트리밍은 client.chat.stream으로 토큰 단위 출력을 받는다.",
]

emb = client.embeddings.create(model="mistral-embed", inputs=corpus)
doc_vectors = [d.embedding for d in emb.data]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


question = "스트리밍은 어떻게 해?"
q_vec = client.embeddings.create(model="mistral-embed", inputs=[question]).data[0].embedding
top = sorted(range(len(corpus)), key=lambda i: cosine(q_vec, doc_vectors[i]), reverse=True)[:2]
context = "\n".join(corpus[i] for i in top)

response = client.chat.complete(
    model="mistral-large-latest",
    messages=[
        {"role": "system", "content": f"다음 문서만 근거로 답해.\n{context}"},
        {"role": "user", "content": question},
    ],
)
print(response.choices[0].message.content)
```

## 6. 문서 처리 (OCR)

PDF나 이미지를 Markdown 텍스트로 바꿉니다. 파일을 먼저 업로드하고 signed URL로 OCR을 호출하는 흐름입니다.

```python
with open("paper.pdf", "rb") as f:
    uploaded = client.files.upload(file={"file_name": "paper.pdf", "content": f})

signed = client.files.get_signed_url(file_id=uploaded.id)
result = client.ocr.process(
    model="mistral-ocr-latest",
    document={"type": "document_url", "document_url": signed.url},
)
for page in result.pages:
    print(page.markdown)
```

## 7. Vibe(코딩 에이전트)로 확장하기

[Vibe](https://mistral.ai/)는 저장소를 직접 읽고 고치는 코딩 에이전트라서, 이 글의 예제 코드를 실제 프로젝트로 키우는 데 딱입니다.

- 저장소를 먼저 지정해서 요청합니다: "이 저장소를 클론해서 예제 코드를 내 프로젝트에 맞게 바꿔 줘"처럼 대상을 분명히 합니다.
- "코드를 고쳐 줘", "리뷰만 해 줘", "테스트를 추가해 줘"처럼 수정·검토·실행을 나누어 의뢰하면결과가安定적입니다.
- 저장소의 `AGENTS.md`나 `CONTRIBUTING.md`를 먼저 읽도록 요청하면 규칙을 자연스럽게 따릅니다.
- API 키는 환경변수로만 쓰고, "비밀키는 절대 코드에 넣지 마"라고 못 박아 두세요.
- 변경은 브랜치에서 커밋하고 PR로 올리도록 요청하면 이력이 깔끔합니다.
- "예제를 실행해서 결과를 확인해 줘", "린트와 타입체크를 돌려 줘"처럼 검증까지 맡기면 완성도가 확 올라갑니다.

## 8. 모델 한눈에 보기

| 모델 | 용도 |
| --- | --- |
| `mistral-large-latest` | 복잡한 추론·코딩·멀티스텝 작업 |
| `mistral-small-latest` | 빠르고 저렴한 일상 대화·분류 |
| `codestral-latest` | 코드 생성·이해 |
| `pixtral-large-latest` | 이미지 입력(vision) |
| `mistral-embed` | 임베딩·검색·RAG |
| `mistral-ocr-latest` | PDF·이미지 문서 파싱 |

요금과 최신 모델 목록은 [La Plateforme](https://console.mistral.ai/)과 [가격표](https://mistral.ai/pricing)에서 확인하세요.

## 참고 자료

- [Mistral AI 공식 문서](https://docs.mistral.ai/)
- [공식 Cookbook 모음](https://docs.mistral.ai/resources/cookbooks)
- [Learn with Mistral](https://learn.mistral.ai/)
- [mistralai Python SDK (GitHub)](https://github.com/mistralai/client-python)

실제 API 사용량에 따른課金은 La Plateforme의請求 정책을 따릅니다.Happy hacking!
