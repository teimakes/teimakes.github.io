from client import client

agent = client.agents.create(
    model="mistral-large-latest",
    name="korean-qa-assistant",
    description="한국어 질문에 답하는 조수 에이전트",
    instructions="항상 한국어 존댓말로, 두 문단 이내로 답한다.",
)

try:
    response = client.agents.complete(
        agent_id=agent.id,
        messages=[{"role": "user", "content": "에이전트가 뭐야? 간단히 설명해 줘."}],
    )
    print(response.choices[0].message.content)
finally:
    client.agents.delete(agent_id=agent.id)
    print(f"에이전트 삭제됨: {agent.id}")
