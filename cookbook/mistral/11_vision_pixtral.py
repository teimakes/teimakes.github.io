import json
import sys

from client import client

if len(sys.argv) < 2:
    print("사용법: python 11_vision_pixtral.py <이미지파일>")
    sys.exit(1)

path = sys.argv[1]

with open(path, "rb") as f:
    uploaded = client.files.upload(file={"file_name": path.split("/")[-1], "content": f})
signed = client.files.get_signed_url(file_id=uploaded.id)

schema = {
    "type": "object",
    "properties": {
        "설명": {"type": "string"},
        "텍스트": {"type": "array", "items": {"type": "string"}},
        "색상": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["설명"],
}

response = client.chat.complete(
    model="pixtral-large-latest",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "이 이미지를 한국어로 설명하고, 보이는 텍스트와 주요 색상을 JSON으로 정리해 줘."},
                {"type": "image_url", "image_url": {"url": signed.url}},
            ],
        }
    ],
    response_format={"type": "json_schema", "json_schema": {"name": "image_desc", "schema": schema, "strict": True}},
)
print(json.dumps(json.loads(response.choices[0].message.content), ensure_ascii=False, indent=2))
