import json
import sys

from client import client

if len(sys.argv) < 2:
    print("사용법: python 10_ocr_structured.py <영수증/문서 이미지|PDF>")
    sys.exit(1)

path = sys.argv[1]

with open(path, "rb") as f:
    uploaded = client.files.upload(file={"file_name": path.split("/")[-1], "content": f})

signed = client.files.get_signed_url(file_id=uploaded.id)
ocr = client.ocr.process(
    model="mistral-ocr-latest",
    document={"type": "document_url", "document_url": signed.url},
)
text = "\n".join(page.markdown for page in ocr.pages)

schema = {
    "type": "object",
    "properties": {
        "상호명": {"type": "string"},
        "날짜": {"type": "string"},
        "품목": {"type": "array", "items": {"type": "object", "properties": {"이름": {"type": "string"}, "가격": {"type": "number"}}}},
        "총액": {"type": "number"},
    },
    "required": ["상호명", "날짜", "총액"],
}

response = client.chat.complete(
    model="mistral-large-latest",
    messages=[
        {"role": "system", "content": "아래 OCR 텍스트에서 영수증 정보를 추출해 JSON으로 반환해."},
        {"role": "user", "content": text},
    ],
    response_format={"type": "json_schema", "json_schema": {"name": "receipt", "schema": schema, "strict": True}},
)
print(json.dumps(json.loads(response.choices[0].message.content), ensure_ascii=False, indent=2))
