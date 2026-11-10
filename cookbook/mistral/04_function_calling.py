import json

from client import client

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "특정 도시의 현재 날씨를 조회한다",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "도시 이름 (예: 서울)"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "수학 수식을 계산한다",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "예: 3+4*2"}},
                "required": ["expression"],
            },
        },
    },
]


def get_weather(city: str) -> str:
    table = {"서울": "맑음, 12도", "부산": "흐림, 15도", "제주": "비, 17도"}
    return table.get(city, "날씨 정보 없음")


def calculate(expression: str) -> str:
    if not set(expression) <= set("0123456789+-*/.() "):
        return "허용되지 않은 문자입니다"
    return str(eval(expression))


functions = {"get_weather": get_weather, "calculate": calculate}

messages = [{"role": "user", "content": "서울 날씨 알려 주고, (3+4)*2 도 계산해 줘."}]

for _ in range(5):
    response = client.chat.complete(
        model="mistral-large-latest", messages=messages, tools=tools, tool_choice="auto"
    )
    message = response.choices[0].message
    if not message.tool_calls:
        print(message.content)
        break
    messages.append(message.model_dump())
    for call in message.tool_calls:
        args = json.loads(call.function.arguments)
        result = functions[call.function.name](**args)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": result,
            }
        )
