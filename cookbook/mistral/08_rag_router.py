import json
import math

from client import client

sources = {
    "menu": [
        "아메리카노 4,500원. 에스프레소 기반의 산뜻한 커피.",
        "카페라떼 5,000원. 우유가 들어가 부드러운 맛.",
        "바닐라라떼 5,500원. 달콤한 바닐라 시럽이 들어간 라떼.",
    ],
    "영업시간": [
        "평일 영업시간은 08:00~22:00, 주말은 10:00~20:00입니다.",
        "공휴일은 10:00~18:00에 운영하며 명절 당일은 휴무입니다.",
    ],
}

vectors = {}
for name, docs in sources.items():
    vectors[name] = [d.embedding for d in client.embeddings.create(model="mistral-embed", inputs=docs).data]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def search(source: str, query: str) -> str:
    docs = sources[source]
    q = client.embeddings.create(model="mistral-embed", inputs=[query]).data[0].embedding
    best = max(range(len(docs)), key=lambda i: cosine(q, vectors[source][i]))
    return docs[best]


tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "메뉴 또는 영업시간 문서에서 검색한다",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "enum": ["menu", "영업시간"]},
                    "query": {"type": "string"},
                },
                "required": ["source", "query"],
            },
        },
    }
]

messages = [{"role": "user", "content": "주말에 언제까지 열어요? 그리고 바닐라라떼 가격 알려 줘."}]

for _ in range(5):
    response = client.chat.complete(
        model="mistral-large-latest", messages=messages, tools=tools, tool_choice="auto"
    )
    message = response.choices[0].message
    if not message.tool_calls:
        print(message.content)
        break
    messages.append(message.model_dump())
    for call in message.tool_calls:
        args = json.loads(call.function.arguments)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": search(**args),
            }
        )
