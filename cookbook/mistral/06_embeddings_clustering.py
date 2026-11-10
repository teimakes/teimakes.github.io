import math
import random

from client import client

sentences = [
    "고양이가 창밖을 바라본다",
    "강아지가 공을 물고 달린다",
    "반려동물이 소파에서 잔다",
    "주식 시장이 오늘 상승했다",
    "금리가 인상되면 채권 가격이 하락한다",
    "중앙은행이 기준금리를 동결했다",
    "사과와 바나나는 과일이다",
    "포도주스는 포도로 만든다",
    "과일 샐러드를 만들었다",
    "파이썬으로 데이터 분석을 했다",
    "머신러닝 모델을 학습시켰다",
    "딥러닝 프레임워크를 비교했다",
]

vectors = [d.embedding for d in client.embeddings.create(model="mistral-embed", inputs=sentences).data]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def kmeans(data, k, iters=30):
    centroids = random.sample(data, k)
    assignments = [0] * len(data)
    for _ in range(iters):
        assignments = [max(range(k), key=lambda j: cosine(v, centroids[j])) for v in data]
        new_centroids = []
        for j in range(k):
            members = [v for v, a in zip(data, assignments) if a == j]
            if members:
                new_centroids.append([sum(col) / len(members) for col in zip(*members)])
            else:
                new_centroids.append(random.choice(data))
        centroids = new_centroids
    return assignments


assignments = kmeans(vectors, k=3)
for j in range(3):
    print(f"클러스터 {j}:")
    for text, a in zip(sentences, assignments):
        if a == j:
            print(f"  - {text}")
