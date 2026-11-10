import math

from client import client

label_examples = {
    "스팸": ["지금 바로 클릭하세요!", "대출 승인 100%", "무료 선물 드립니다"],
    "일상": ["오늘 점심 뭐 먹지", "내일 회의 시간 확인해 줘", "고양이가 소파에서 잔다"],
    "업무": ["다음 분기 보고서 제출 기한 확인", "회의 일정 조율 부탁드립니다", "프로젝트 일정 변경 안내"],
}

texts = ["회의를 오후 3시로 옮길 수 있을까요?", "공짜 아이폰 당첨! 링크 클릭", "점심으로 김치찌개 먹었다"]

flat_examples = [(label, text) for label, items in label_examples.items() for text in items]
flat_texts = [text for _, text in flat_examples]

vectors = [d.embedding for d in client.embeddings.create(model="mistral-embed", inputs=flat_texts).data]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


for text in texts:
    q = client.embeddings.create(model="mistral-embed", inputs=[text]).data[0].embedding
    best = max(range(len(flat_texts)), key=lambda i: cosine(q, vectors[i]))
    print(f"{text}  →  {flat_examples[best][0]}")
