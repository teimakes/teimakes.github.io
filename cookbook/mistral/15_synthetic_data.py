import argparse
import json

from client import client

parser = argparse.ArgumentParser()
parser.add_argument("--count", type=int, default=20)
parser.add_argument("--finetune", action="store_true", help="생성한 JSONL로 fine-tuning 잡 생성")
args = parser.parse_args()

schema = {
    "type": "object",
    "properties": {
        "samples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["user", "assistant"]},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                    }
                },
                "required": ["messages"],
            },
        }
    },
    "required": ["samples"],
}

response = client.chat.complete(
    model="mistral-large-latest",
    messages=[
        {
            "role": "user",
            "content": (
                f"고객 응대 챗봇 학습용 합성 대화 {args.count}개를 만들어 줘. "
                "주제는 제품 문의, 반품, 배송, 결제 오류 등 일상적인 고객 서비스 상황으로 다양하게. "
                "각 샘플은 user와 assistant 메시지 쌍의 배열로 반환해."
            ),
        }
    ],
    response_format={"type": "json_schema", "json_schema": {"name": "synthetic", "schema": schema, "strict": True}},
)
samples = json.loads(response.choices[0].message.content)["samples"]

with open("synthetic_data.jsonl", "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps({"messages": s["messages"]}, ensure_ascii=False) + "\n")

print(f"synthetic_data.jsonl에 {len(samples)}개 샘플 저장")

if args.finetune:
    with open("synthetic_data.jsonl", "rb") as f:
        uploaded = client.files.upload(
            file={"file_name": "synthetic_data.jsonl", "content": f}, purpose="fine-tune"
        )
    job = client.fine_tuning.jobs.create(
        training_files={"file_id": uploaded.id},
        model="mistral-small-latest",
    )
    print(f"fine-tuning 잡 생성됨: {job.id} (상태: {job.status})")
