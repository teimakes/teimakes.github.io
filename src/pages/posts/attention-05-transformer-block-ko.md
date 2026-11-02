---
layout: '../../layouts/Post.astro'
title: '어텐션 이해하기 ⑤: 작은 트랜스포머 디코더 직접 만들기'
description: '토큰·위치 임베딩부터 인과적 멀티헤드 어텐션, FFN, 다음 토큰 확률까지 한 파일로 연결합니다.'
date: '2026-11-03'
tags: ['attention', 'transformer', 'llm', 'python']
---

> **어텐션과 트랜스포머 시리즈 · 5/5**  
> 큰 그림부터 실행 코드까지 차근차근 살펴봅니다. 각 글의 코드는 독립 실행 가능합니다.

## 시리즈 목차

1. [어텐션 이해하기 ①: 큰 그림부터 가중합까지](/posts/attention-01-big-picture-ko/)
2. [어텐션 이해하기 ②: Q·K·V와 스케일드 닷 프로덕트](/posts/attention-02-qkv-ko/)
3. [어텐션 이해하기 ③: GPT는 왜 미래 토큰을 가릴까?](/posts/attention-03-causal-mask-ko/)
4. [어텐션 이해하기 ④: 멀티헤드와 텐서 모양 추적하기](/posts/attention-04-multi-head-ko/)
5. [어텐션 이해하기 ⑤: 작은 트랜스포머 디코더 직접 만들기](/posts/attention-05-transformer-block-ko/)

**실행 환경:** Python 3와 NumPy가 필요합니다. 터미널에서 `python -m pip install numpy`를 실행한 뒤, 아래 코드를 하나의 `.py` 파일로 저장해 실행하거나 노트북 셀에 붙여 넣으세요.

## 이번에는 블록 전체를 연결합니다

지금까지 살펴본 어텐션은 트랜스포머의 한 부분입니다. 이번 글에서는 다음 흐름을 하나의 실행 파일로 연결합니다.

**토큰 ID → 토큰 임베딩 + 위치 임베딩 → 정규화 → 인과적 멀티헤드 어텐션 → 잔차 연결 → 정규화 → FFN → 잔차 연결 → 출력 정규화 → 어휘 점수**

이 구현은 이해를 위한 **작고 학습되지 않은 디코더형 블록**입니다. 실제 GPT 구현과 달리 편향, 학습 가능한 LayerNorm 스케일·시프트, 드롭아웃, 여러 층, 토크나이저, 학습 루프, KV 캐시 등을 생략했습니다. 따라서 출력 확률은 임의 값이고 문장의 의미를 이해한 결과가 아닙니다.

모양은 토큰 수 `T=3`, 표현 차원 `D=4`, 헤드 수 `H=2`, 어휘 수 `4`입니다. 배치 차원은 생략했습니다.

## 완전한 코드: NumPy로 구현한 단일 디코더 블록

```python
import numpy as np

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(11)
vocab = ["나는", "사과를", "먹는다", "본다"]
ids = np.array([0, 1, 2])
T, D, H = len(ids), 4, 2
Dh = D // H

def init(shape):
    return rng.normal(0.0, 0.2, shape)

def softmax(x):
    ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)

def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

token_embedding = init((len(vocab), D))
position_embedding = init((T, D))
Wq, Wk, Wv, Wo = (init((D, D)) for _ in range(4))
W1, W2 = init((D, 2 * D)), init((2 * D, D))
W_vocab = init((D, len(vocab)))

def decoder(token_ids):
    length = len(token_ids)
    x = token_embedding[token_ids] + position_embedding[:length]
    z = layer_norm(x)

    def heads(matrix):
        return matrix.reshape(length, H, Dh).transpose(1, 0, 2)

    q, k, v = heads(z @ Wq), heads(z @ Wk), heads(z @ Wv)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(Dh)
    future = np.triu(np.ones((length, length), dtype=bool), k=1)
    weights = softmax(np.where(future[None, :, :], -np.inf, scores))
    context = weights @ v
    merged = context.transpose(1, 0, 2).reshape(length, D)
    x = x + merged @ Wo

    z = layer_norm(x)
    feed_forward = np.maximum(0.0, z @ W1) @ W2
    x = x + feed_forward
    logits = layer_norm(x) @ W_vocab
    return logits, weights

logits, weights = decoder(ids)
next_prob = softmax(logits[-1])
changed_logits, _ = decoder(np.array([0, 1, 3]))

print("입력:", [vocab[i] for i in ids])
print("logits 모양:", logits.shape)
for word, prob in zip(vocab, next_prob):
    print(f"{word}: {prob:.3f}")
print("미래 변경 후 앞 두 logits 동일:", np.allclose(logits[:2], changed_logits[:2]))
assert np.allclose(next_prob.sum(), 1.0)
assert np.allclose(logits[:2], changed_logits[:2])
```

### 실행 결과

```text
입력: ['나는', '사과를', '먹는다']
logits 모양: (3, 4)
나는: 0.311
사과를: 0.284
먹는다: 0.174
본다: 0.232
미래 변경 후 앞 두 logits 동일: True
```

## 한 줄씩 따라가기: 준비 단계

- `vocab`은 네 개의 가상 토큰 목록이고 `ids`는 입력 토큰의 정수 ID입니다. 실제 토큰화는 구현하지 않습니다.
- `T, D, H`와 `Dh`는 입력 길이, 모델 차원, 헤드 수, 헤드당 차원을 정합니다.
- `init(shape)`은 난수 파라미터를 만듭니다. 고정 시드 덕분에 같은 환경에서는 결과를 재현할 수 있습니다.
- `softmax(x)`는 마지막 축을 확률로 정규화합니다. 계산 전 최대값을 빼 오버플로를 줄입니다.
- `layer_norm(x)`는 각 토큰의 마지막 차원에 대해 평균과 분산을 계산하고 정규화합니다. `eps`는 0으로 나누는 일을 방지합니다. 학습 가능한 스케일과 편향은 생략했습니다.
- `token_embedding`은 어휘별 벡터, `position_embedding`은 위치별 벡터입니다. 예제에서는 둘 다 난수입니다.
- `Wq`부터 `Wo`까지는 어텐션 투영 행렬입니다. `W1`, `W2`는 FFN, `W_vocab`은 어휘 점수를 만드는 행렬입니다.

## 한 줄씩 따라가기: 디코더 함수

- `length = len(token_ids)`는 이번 입력 길이입니다. 이 장난감 구현은 미리 만들어 둔 위치 임베딩 길이를 초과하는 입력을 지원하지 않습니다.
- `token_embedding[token_ids] + position_embedding[:length]`는 토큰 벡터와 각 위치 벡터를 더합니다. 결과 모양은 `(T, D)`입니다.
- `z = layer_norm(x)`는 어텐션 전에 각 토큰을 정규화합니다. 이 예제는 **pre-norm** 배치를 택했습니다.
- 내부 함수 `heads`는 `(T, D)`를 `(H, T, Dh)`로 바꿉니다.
- `q, k, v = ...`는 정규화된 입력을 각 투영 행렬에 통과시키고 헤드를 나눕니다.
- `scores = ...`는 헤드별 Q·K 내적을 구한 후 `sqrt(Dh)`로 나눕니다.
- `future = ...`와 `np.where(...)`는 대각선 위의 미래 토큰을 가리고 소프트맥스로 가중치를 계산합니다.
- `context = weights @ v`는 각 헤드의 값 벡터를 가중합합니다.
- `transpose(...).reshape(...)`는 헤드별 결과를 다시 토큰별로 결합합니다.
- `x = x + merged @ Wo`는 출력 투영에 원래 입력을 더하는 **잔차 연결**입니다.
- 두 번째 `layer_norm(x)`는 FFN 입력을 정규화합니다.
- `np.maximum(0.0, z @ W1) @ W2`는 첫 선형 변환, ReLU 활성화, 두 번째 선형 변환으로 이루어진 위치별 FFN입니다. 실제 모델은 다른 활성화 함수나 FFN 구조를 사용할 수 있습니다.
- `x = x + feed_forward`는 FFN 결과에 대한 두 번째 잔차 연결입니다.
- `layer_norm(x) @ W_vocab`은 각 위치의 표현을 어휘 크기의 점수, 즉 **logits**로 바꿉니다. logits 자체는 확률이 아닙니다.
- `return logits, weights`는 출력과 헤드별 참고 비율을 함께 돌려줍니다.

## 한 줄씩 따라가기: 출력과 검증

- `decoder(ids)`는 각 입력 위치에 대한 logits를 계산합니다.
- `softmax(logits[-1])`는 마지막 위치의 logits를 다음 토큰 후보의 확률로 바꿉니다. 난수 가중치이므로 이 확률에는 의미 있는 예측력이 없습니다.
- `changed_logits, _ = ...`는 마지막 입력만 바꾸어 다시 계산합니다.
- 첫 번째 `assert`는 확률의 합이 1인지 확인하고, 두 번째는 미래 입력을 바꾸더라도 앞 두 위치의 logits가 변하지 않는지 검사합니다.

## 실제 학습에서는 무엇이 추가될까?

1. 충분한 텍스트를 토큰화해 학습용 시퀀스를 만듭니다.
2. 입력과 정답을 한 칸 어긋나게 만듭니다. 예를 들어 `[A, B, C]`를 입력하면 각 위치의 정답은 `[B, C, 다음 토큰]`입니다.
3. 각 위치의 logits와 정답 ID로 크로스 엔트로피 손실을 계산합니다. 사용하지 않는 패딩 위치 등은 손실 계산에서 제외해야 합니다.
4. 자동 미분과 옵티마이저로 임베딩, Q·K·V, FFN, 출력 행렬을 업데이트합니다.
5. 별도의 검증 데이터에서 손실과 성능을 확인하고, 생성 시에는 확률 분포에서 다음 토큰을 선택해 반복합니다.

이 예제는 NumPy 순전파만 보여 주므로 자동 미분이나 학습을 수행하지 않습니다. 따라서 실제 언어 모델을 학습하려면 PyTorch 같은 프레임워크와 충분한 데이터가 필요합니다.

## 계산 비용과 다음 주제

기본적인 밀집 어텐션의 점수 행렬은 헤드마다 `(T, T)`이므로 시퀀스 길이가 길어질수록 계산량과 명시적 점수 저장량이 빠르게 늘어납니다. 실제 구현에서는 메모리 접근을 최적화하는 방식이나 캐시 등을 사용하며, 세부 비용은 구현에 따라 달라집니다.

**직접 바꿔 보기:** 마지막 토큰 ID를 바꾸고 첫 두 logits가 변하지 않는지 확인해 보세요. 위치 임베딩을 0으로 만들었을 때의 출력도 비교해 볼 수 있지만, 단일 예제의 숫자 변화만으로 모델의 순서 이해 능력을 평가할 수는 없습니다.

## 참고 자료

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 트랜스포머 원 논문
- PyTorch, [Scaled Dot Product Attention 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — 스케일링과 마스킹을 포함한 구현 참고
- David Louapre, [Natural Language Processing before LLMs](https://huggingface.co/spaces/dlouapre/nlp-before-llms) — 어텐션까지 이어지는 자연어 처리의 흐름
