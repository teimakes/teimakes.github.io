import json

from client import client

questions = [
    "한 문장으로 LLM을 설명해 줘.",
    "RAG의 장점을 한 문장으로 말해 줘.",
    "임베딩이 뭐야? 한 문장으로.",
]

lines = []
for i, q in enumerate(questions):
    lines.append(
        json.dumps(
            {
                "custom_id": f"req-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "mistral-small-latest",
                    "messages": [{"role": "user", "content": q}],
                },
            },
            ensure_ascii=False,
        )
    )

with open("batch_input.jsonl", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

with open("batch_input.jsonl", "rb") as f:
    uploaded = client.files.upload(file={"file_name": "batch_input.jsonl", "content": f}, purpose="batch")

job = client.batch.jobs.create(
    input_files={"file_id": uploaded.id},
    model="mistral-small-latest",
    endpoint="/v1/chat/completions",
)
print(f"배치 잡 생성됨: {job.id} (상태: {job.status})")

for j in client.batch.jobs.list():
    print(f"  잡 {j.id}: {j.status}")
