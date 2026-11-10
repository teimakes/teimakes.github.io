from client import client

stream = client.chat.stream(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": "제주도 3박 4일 여행 일정을 짜 줘."}],
)
for chunk in stream:
    delta = chunk.data.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
