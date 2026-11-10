import math

from client import client

documents = [
    "Mistral API는 La Plateforme 콘솔(https://console.mistral.ai/)에서 API 키를 발급받아 사용한다. 키는 환경변수 MISTRAL_API_KEY로 관리하는 것이 안전하다.",
    "mistral-small-latest는 빠르고 저렴해서 프로토타이핑과 일상 대화에 적합하다. 대규모 배치 작업에도 자주 쓰인다.",
    "mistral-large-latest는 복잡한 추론, 코딩, 멀티스텝 에이전트 작업에 강한 플래그십 모델이다.",
    "스트리밍은 client.chat.stream()으로 시작하며, 청크 단위로 토큰을 받아 UI에 실시간 출력할 수 있다.",
    "구조화 출력은 response_format에 JSON Schema를 지정하면 모델이 해당 스키마의 JSON만 반환한다.",
    "임베딩 모델 mistral-embed는 문장을 벡터로 바꿔 검색(RAG)과 분류, 클러스터링에 사용한다.",
    "OCR 모델 mistral-ocr-latest는 PDF나 이미지 문서를 Markdown으로 변환한다.",
    "배치 추론은 client.batch API로 대량 요청을 한 번에 처리하고 비용을 절감한다.",
]


def chunk(text, size=200):
    return [text[i : i + size] for i in range(0, len(text), size)]


chunks = [c for doc in documents for c in chunk(doc)]

emb = client.embeddings.create(model="mistral-embed", inputs=chunks)
chunk_vectors = [d.embedding for d in emb.data]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


question = "스트리밍은 어떻게 해?"
q_vec = client.embeddings.create(model="mistral-embed", inputs=[question]).data[0].embedding
top = sorted(range(len(chunks)), key=lambda i: cosine(q_vec, chunk_vectors[i]), reverse=True)[:3]
context = "\n\n".join(chunks[i] for i in top)

print("검색된 컨텍스트:")
print(context)
print("-" * 40)

response = client.chat.complete(
    model="mistral-large-latest",
    messages=[
        {"role": "system", "content": f"다음 문서만 근거로 답해. 문서에 없으면 모른다고 말해.\n\n{context}"},
        {"role": "user", "content": question},
    ],
)
print(response.choices[0].message.content)
