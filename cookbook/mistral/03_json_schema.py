import json

from client import client

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
print(data["summary"])
