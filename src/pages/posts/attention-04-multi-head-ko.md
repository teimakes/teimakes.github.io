---
layout: '../../layouts/Post.astro'
title: '어텐션 이해하기 ④: 멀티헤드와 텐서 모양 추적하기'
description: '두 개의 인과적 어텐션 헤드를 NumPy로 구현하고 reshape와 transpose의 역할을 살펴봅니다.'
date: '2026-11-03'
tags: ['attention', 'transformer', 'llm', 'python']
---

> **어텐션과 트랜스포머 시리즈 · 4/5**  
> 큰 그림부터 실행 코드까지 차근차근 살펴봅니다. 각 글의 코드는 독립 실행 가능합니다.

## 시리즈 목차

1. [어텐션 이해하기 ①: 큰 그림부터 가중합까지](/posts/attention-01-big-picture-ko/)
2. [어텐션 이해하기 ②: Q·K·V와 스케일드 닷 프로덕트](/posts/attention-02-qkv-ko/)
3. [어텐션 이해하기 ③: GPT는 왜 미래 토큰을 가릴까?](/posts/attention-03-causal-mask-ko/)
4. [어텐션 이해하기 ④: 멀티헤드와 텐서 모양 추적하기](/posts/attention-04-multi-head-ko/)
5. [어텐션 이해하기 ⑤: 작은 트랜스포머 디코더 직접 만들기](/posts/attention-05-transformer-block-ko/)

**실행 환경:** Python 3와 NumPy가 필요합니다. 터미널에서 `python -m pip install numpy`를 실행한 뒤, 아래 코드를 하나의 `.py` 파일로 저장해 실행하거나 노트북 셀에 붙여 넣으세요.

## 왜 헤드가 여러 개일까?

**멀티헤드 어텐션**은 입력을 서로 다른 투영 공간에서 나누어 처리한 결과를 합칩니다. 각 헤드는 서로 다른 가중치로 서로 다른 패턴을 학습할 수 있습니다. 다만 특정 헤드가 반드시 정해진 문법 역할 하나를 전담하는 것은 아닙니다.

이 예제는 `T=3`, `d_model=4`, `헤드 수=2`, `헤드당 차원=2`입니다. 배치 차원은 이해를 위해 생략합니다. 가중치는 고정 난수로 초기화되며 학습하지 않습니다.

## 모양 지도

- X: `(3, 4)`
- X @ Wq: `(3, 4)` → 분할 후 Q: `(2, 3, 2)`
- K 전치: `(2, 2, 3)` → 점수: `(2, 3, 3)`
- 헤드별 출력: `(2, 3, 2)` → 결합 후: `(3, 4)`
- 최종 출력: `(3, 4)`

앞의 `2`는 헤드 수, `3`은 토큰 수입니다. 실제 학습 코드는 보통 맨 앞에 배치 차원 `B`를 추가합니다.

## 완전한 코드: 두 개의 인과적 헤드

```python
import numpy as np

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(7)
X = np.array([[1., 0., 1., 0.],
              [0., 1., 0., 1.],
              [1., 1., 0., 0.]])
T, d_model = X.shape
n_heads = 2
d_head = d_model // n_heads

Wq = rng.normal(0, 0.3, (d_model, d_model))
Wk = rng.normal(0, 0.3, (d_model, d_model))
Wv = rng.normal(0, 0.3, (d_model, d_model))
Wo = rng.normal(0, 0.3, (d_model, d_model))

def split_heads(x):
    return x.reshape(T, n_heads, d_head).transpose(1, 0, 2)

Q = split_heads(X @ Wq)
K = split_heads(X @ Wk)
V = split_heads(X @ Wv)
scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_head)
future = np.triu(np.ones((T, T), dtype=bool), k=1)
scores = np.where(future[None, :, :], -np.inf, scores)
exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
heads = weights @ V
merged = heads.transpose(1, 0, 2).reshape(T, d_model)
output = merged @ Wo

print("Q 모양:", Q.shape)
print("어텐션 가중치 모양:", weights.shape)
print("첫 번째 헤드 가중치:\n", weights[0])
print("두 번째 헤드 가중치:\n", weights[1])
print("출력 모양:", output.shape)
assert np.allclose(weights.sum(axis=-1), 1)
assert np.all(weights[:, np.triu_indices(T, 1)[0], np.triu_indices(T, 1)[1]] == 0)
```

### 실행 결과

```text
Q 모양: (2, 3, 2)
어텐션 가중치 모양: (2, 3, 3)
첫 번째 헤드 가중치:
 [[1.    0.    0.   ]
 [0.512 0.488 0.   ]
 [0.327 0.326 0.347]]
두 번째 헤드 가중치:
 [[1.    0.    0.   ]
 [0.447 0.553 0.   ]
 [0.33  0.338 0.332]]
출력 모양: (3, 4)
```

## 한 줄씩 따라가기

- `default_rng(7)`은 재현을 위한 난수 생성기입니다. 가중치가 고정된 임의 값이라는 뜻이지, 학습된 값이라는 뜻은 아닙니다.
- `X`의 세 행은 세 토큰, 네 열은 각 토큰의 표현 차원입니다.
- `T, d_model = X.shape`는 토큰 수와 모델 차원을 읽습니다.
- `n_heads = 2`, `d_head = d_model // n_heads`는 헤드당 차원을 2로 정합니다. 이 예제는 모델 차원이 헤드 수로 나누어진다고 가정합니다.
- 네 개의 `rng.normal(...)`은 Q·K·V와 최종 출력 투영 행렬을 만듭니다.
- `reshape(T, n_heads, d_head)`는 각 위치의 4차원을 두 헤드의 2차원으로 나눕니다.
- `transpose(1, 0, 2)`는 축 순서를 `(헤드, 토큰, 차원)`으로 바꿉니다. 데이터를 뒤섞어 학습시키는 작업이 아니라 축의 해석 순서를 변경합니다.
- `Q`, `K`, `V`는 각각 `(2, 3, 2)`입니다. 하나의 큰 투영 행렬을 헤드별 슬라이스로 나누어 별도 투영을 한 것과 같은 형태로 구성합니다.
- `K.transpose(0, 2, 1)`은 헤드 축을 유지하면서 토큰과 차원 축을 바꿉니다. NumPy의 배치 행렬곱으로 헤드별 `(3, 3)` 점수를 계산합니다.
- `future`는 대각선 위의 미래 위치입니다. `future[None, :, :]`는 헤드 축에 브로드캐스팅해 모든 헤드에 같은 마스크를 적용합니다.
- `exp_scores`, `weights`는 각 헤드의 각 query 행별로 소프트맥스를 구합니다.
- `heads = weights @ V`는 헤드별 문맥을 계산합니다.
- `transpose(...).reshape(...)`는 헤드 출력을 토큰별로 이어 붙여 `(3, 4)`로 복원합니다.
- `output = merged @ Wo`는 결합된 결과를 다시 투영합니다.
- 출력과 `assert`는 헤드별 가중치의 행 합, 미래 차단, 최종 모양을 확인합니다.

## 결과 읽기

첫 행은 자기 자신만 볼 수 있어 두 헤드 모두 `[1, 0, 0]`입니다. 나머지 행의 가중치는 가중치 행렬이 달라 서로 다를 수 있습니다. 여기서는 난수 가중치이므로 차이가 의미 있는 언어적 관계를 학습했다는 증거는 아닙니다.

**직접 바꿔 보기:** 난수 시드를 바꾸거나 첫 입력 벡터를 수정해 헤드별 가중치를 비교하세요. `d_model`이 헤드 수로 나누어떨어지지 않으면 이 구현의 `reshape`가 성립하지 않습니다.

다음 글에서는 어텐션을 잔차 연결, 정규화, FFN과 연결해 작은 디코더 블록을 만듭니다.

## 참고 자료

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 트랜스포머 원 논문
- PyTorch, [Scaled Dot Product Attention 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — 스케일링과 마스킹을 포함한 구현 참고
- David Louapre, [Natural Language Processing before LLMs](https://huggingface.co/spaces/dlouapre/nlp-before-llms) — 어텐션까지 이어지는 자연어 처리의 흐름
