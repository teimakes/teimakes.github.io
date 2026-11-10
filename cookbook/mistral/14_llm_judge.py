import json

from client import client

question = "RAG가 뭐야?"
answer_a = "RAG는 검색으로 문서에서 관련 내용을 찾아 LLM에 컨텍스트로 넣어 답변 정확도를 높이는 방식입니다."
answer_b = "RAG는 그냥 LLM을 크게 만드는 기술이에요."

schema = {
    "type": "object",
    "properties": {
        "승자": {"type": "string", "enum": ["A", "B", "동점"]},
        "점수_A": {"type": "integer", "minimum": 0, "maximum": 10},
        "점수_B": {"type": "integer", "minimum": 0, "maximum": 10},
        "근거": {"type": "string"},
    },
    "required": ["승자", "점수_A", "점수_B", "근거"],
}

response = client.chat.complete(
    model="mistral-large-latest",
    messages=[
        {
            "role": "user",
            "content": (
                f"질문: {question}\n\n"
                f"답변 A: {answer_a}\n\n"
                f"답변 B: {answer_b}\n\n"
                "두 답변을 정확성 기준으로 0~10점으로 채점하고 승자를 JSON으로 반환해 줘."
            ),
        }
    ],
    response_format={"type": "json_schema", "json_schema": {"name": "judge", "schema": schema, "strict": True}},
)
verdict = json.loads(response.choices[0].message.content)
print(json.dumps(verdict, ensure_ascii=False, indent=2))
