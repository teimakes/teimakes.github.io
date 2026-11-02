---
layout: '../../layouts/Post.astro'
title: '어텐션 이해하기 ②: Q·K·V와 스케일드 닷 프로덕트'
description: '토큰 벡터로부터 Q, K, V를 만들고 점수 행렬과 어텐션 출력을 직접 계산합니다.'
date: '2026-11-03'
tags: ['attention', 'transformer', 'llm', 'python']
---

> **어텐션과 트랜스포머 시리즈 · 2/5**  
> 큰 그림부터 실행 코드까지 차근차근 살펴봅니다. 각 글의 코드는 독립 실행 가능합니다.

## 시리즈 목차

1. [어텐션 이해하기 ①: 큰 그림부터 가중합까지](/posts/attention-01-big-picture-ko/)
2. [어텐션 이해하기 ②: Q·K·V와 스케일드 닷 프로덕트](/posts/attention-02-qkv-ko/)
3. [어텐션 이해하기 ③: GPT는 왜 미래 토큰을 가릴까?](/posts/attention-03-causal-mask-ko/)
4. [어텐션 이해하기 ④: 멀티헤드와 텐서 모양 추적하기](/posts/attention-04-multi-head-ko/)
5. [어텐션 이해하기 ⑤: 작은 트랜스포머 디코더 직접 만들기](/posts/attention-05-transformer-block-ko/)

**실행 환경:** Python 3와 NumPy가 필요합니다. 터미널에서 `python -m pip install numpy`를 실행한 뒤, 아래 코드를 하나의 `.py` 파일로 저장해 실행하거나 노트북 셀에 붙여 넣으세요.

## 점수는 어디에서 나올까?

앞 글에서는 점수를 직접 지정했습니다. 실제 셀프 어텐션에서는 입력 벡터 `X`를 서로 다른 가중치 행렬로 변환해 세 표현을 만듭니다.

- **Query(Q):** 현재 위치가 무엇을 참고할지 비교하는 표현
- **Key(K):** 각 위치가 비교에 사용될 수 있도록 만든 표현
- **Value(V):** 가중합해 전달할 정보

도서관의 검색어·색인·자료에 비유할 수 있지만, 실제 Q·K·V는 사람이 작성한 문장이 아니라 **학습되는 수치 표현**입니다. 셀프 어텐션에서는 셋 모두 같은 입력 `X`에서 출발합니다. 서로 다른 입력에서 Q와 K·V를 만드는 크로스 어텐션도 있습니다.

핵심 계산은 `softmax(Q @ K.T / sqrt(d_k)) @ V`입니다. 각 행이 하나의 query 위치이고 각 열이 참고할 key 위치입니다.

## 데이터 모양부터 확인하기

이 글은 배치 차원을 생략합니다. 토큰 수 `T=3`, 모델 차원 `D=4`, Q·K 차원 `d_k=2`, V 차원 `d_v=2`인 작은 예제를 사용합니다.

- X: `(3, 4)` / Wq, Wk, Wv: 각각 `(4, 2)`
- Q, K, V: 각각 `(3, 2)` / K.T: `(2, 3)`
- 점수와 가중치: `(3, 3)` / 최종 출력: `(3, 2)`

## 완전한 코드: 직접 만드는 셀프 어텐션

행렬은 계산 결과를 살펴보기 위한 수동 설정값이며, 학습된 모델의 가중치가 아닙니다. 마스킹을 적용하지 않으므로 각 위치가 모든 위치를 볼 수 있습니다.

```python
import numpy as np

np.set_printoptions(precision=3, suppress=True)

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

X = np.array([[1., 0., 1., 0.],
              [0., 1., 0., 1.],
              [1., 1., 0., 0.]])
Wq = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
Wk = np.array([[1., 0.], [0., 1.], [0., 0.], [0., 0.]])
Wv = np.array([[1., 0.], [0., 1.], [1., 1.], [0., 1.]])

Q, K, V = X @ Wq, X @ Wk, X @ Wv
scores = Q @ K.T / np.sqrt(K.shape[-1])
weights = softmax(scores)
output = weights @ V

print("Q:\n", Q, "\nK:\n", K, "\nV:\n", V)
print("점수:\n", scores)
print("가중치:\n", weights)
print("출력:\n", output)
assert np.allclose(weights.sum(axis=-1), 1.0)
```

### 실행 결과

```text
Q:
 [[2. 0.]
 [0. 2.]
 [1. 1.]] 
K:
 [[1. 0.]
 [0. 1.]
 [1. 1.]] 
V:
 [[2. 1.]
 [0. 2.]
 [1. 1.]]
점수:
 [[1.414 0.    1.414]
 [0.    1.414 1.414]
 [0.707 0.707 1.414]]
가중치:
 [[0.446 0.108 0.446]
 [0.108 0.446 0.446]
 [0.248 0.248 0.503]]
출력:
 [[1.337 1.108]
 [0.663 1.446]
 [1.    1.248]]
```

## 한 줄씩 따라가기

- `softmax`의 첫 줄은 각 행의 최댓값을 빼서 수치적으로 안정적인 지수 연산을 준비합니다. `axis=-1`은 마지막 축, 즉 key 위치별 계산을 뜻합니다.
- `np.exp(shifted)`는 각 점수를 양수로 변환하고, 다음 줄은 행마다 합으로 나눕니다. 각 query가 모든 key에 나누어 준 가중치의 합은 1입니다.
- `X = ...`의 각 행은 하나의 토큰 벡터입니다. 세 행을 한꺼번에 계산합니다.
- `Wq`, `Wk`, `Wv`는 동일한 입력에서 서로 다른 Q·K·V를 만드는 투영 행렬입니다. 실제 모델에서는 학습 중 이 값들이 업데이트됩니다.
- `Q, K, V = X @ Wq, ...`: 각 토큰에서 Q·K·V를 만듭니다. 예를 들어 Q의 첫 행은 X 첫 행과 Wq의 행렬곱입니다.
- `Q @ K.T`: 각 query와 각 key의 내적을 한 번에 계산합니다. `(3, 2) @ (2, 3)`의 결과는 `(3, 3)`입니다.
- `/ np.sqrt(K.shape[-1])`: key 차원의 제곱근으로 점수를 나눕니다. 차원이 커질 때 내적의 규모가 지나치게 커져 소프트맥스가 과도하게 뾰족해지는 현상을 완화합니다.
- `weights = softmax(scores)`: 비교 점수를 각 query의 참고 비율로 바꿉니다.
- `output = weights @ V`: `(3, 3) @ (3, 2)`의 결과로 query마다 길이 2인 문맥 표현을 얻습니다.
- `print(...)`는 중간 결과를 보여 주고 `assert`는 가중치의 행별 합이 1인지 확인합니다.

## 첫 번째 query를 손으로 확인하기

출력에서 첫 번째 Q는 `[2, 0]`입니다. 첫 번째 K `[1, 0]`과의 내적은 2이고, 두 번째 K `[0, 1]`과의 내적은 0입니다. 각각을 `sqrt(2)`로 나누고 세 점수에 소프트맥스를 적용합니다. 그 결과가 첫 행의 가중치이고, 이를 V 세 행에 곱해 더하면 출력의 첫 행이 됩니다.

**직접 바꿔 보기:** 첫 번째 입력 벡터나 Wq의 값을 바꾸고 점수·가중치가 어떻게 달라지는지 살펴보세요. 같은 X를 사용하더라도 투영 행렬이 달라 Q, K, V는 서로 다를 수 있습니다.

## 스스로 확인하기

1. 점수 행렬에서 행과 열은 각각 무엇을 뜻하나요?
2. Q와 K의 마지막 차원이 같아야 하는 이유는 무엇일까요?
3. 가중치의 열 합이 반드시 1일까요, 아니면 행 합이 1일까요?

다음 글에서는 미래 위치를 가리는 인과적 마스크를 추가합니다.

## 참고 자료

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 트랜스포머 원 논문
- PyTorch, [Scaled Dot Product Attention 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — 스케일링과 마스킹을 포함한 구현 참고
- David Louapre, [Natural Language Processing before LLMs](https://huggingface.co/spaces/dlouapre/nlp-before-llms) — 어텐션까지 이어지는 자연어 처리의 흐름
