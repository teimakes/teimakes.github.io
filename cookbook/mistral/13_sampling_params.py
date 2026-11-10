from client import client

prompt = "한국의 위대한 발명품 하나를 한 문장으로 소개해 줘."

configs = [
    {"temperature": 0.0, "top_p": 1.0},
    {"temperature": 0.7, "top_p": 0.9},
    {"temperature": 1.2, "top_p": 0.95},
]

for cfg in configs:
    print(f"[temperature={cfg['temperature']}, top_p={cfg['top_p']}]")
    for _ in range(2):
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
        )
        print(" ", response.choices[0].message.content)
    print("-" * 40)
