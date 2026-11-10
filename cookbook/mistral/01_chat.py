from client import client

messages = [
    {"role": "system", "content": "너는 한국어로 답하는 간결한 조수다. 코드는 코드 블록으로 감싼다."},
    {"role": "user", "content": "RAG가 뭐야? 한 문단으로 설명해 줘."},
]

for model in ("mistral-small-latest", "mistral-large-latest"):
    response = client.chat.complete(model=model, messages=messages)
    print(f"[{model}]")
    print(response.choices[0].message.content)
    print(f"사용 토큰: {response.usage.total_tokens}")
    print("-" * 40)
