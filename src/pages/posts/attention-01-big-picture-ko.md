---
layout: '../../layouts/Post.astro'
title: '어텐션 이해하기 ①: 큰 그림부터 가중합까지'
description: 'LLM에서 어텐션이 필요한 이유와 점수, 소프트맥스, 가중합을 실행 가능한 NumPy 예제로 이해합니다.'
date: '2026-11-03'
tags: ['attention', 'transformer', 'llm', 'python']
---

> **어텐션과 트랜스포머 시리즈 · 1/5**  
> 큰 그림부터 실행 코드까지 차근차근 살펴봅니다. 각 글의 코드는 독립 실행 가능합니다.

## 시리즈 목차

1. [어텐션 이해하기 ①: 큰 그림부터 가중합까지](/posts/attention-01-big-picture-ko/)
2. [어텐션 이해하기 ②: Q·K·V와 스케일드 닷 프로덕트](/posts/attention-02-qkv-ko/)
3. [어텐션 이해하기 ③: GPT는 왜 미래 토큰을 가릴까?](/posts/attention-03-causal-mask-ko/)
4. [어텐션 이해하기 ④: 멀티헤드와 텐서 모양 추적하기](/posts/attention-04-multi-head-ko/)
5. [어텐션 이해하기 ⑤: 작은 트랜스포머 디코더 직접 만들기](/posts/attention-05-transformer-block-ko/)

**실행 환경:** Python 3와 NumPy가 필요합니다. 터미널에서 `python -m pip install numpy`를 실행한 뒤, 아래 코드를 하나의 `.py` 파일로 저장해 실행하거나 노트북 셀에 붙여 넣으세요.

## 이번 글에서 만들 것

LLM이 “나는 사과를 먹는다”를 처리한다고 생각해 봅시다. ‘먹는다’를 해석할 때는 문장 안의 다른 토큰도 참고해야 합니다. 어텐션은 현재 위치가 다른 위치의 정보를 **얼마나 섞어 가져올지** 계산하는 방법입니다.

단어마다 독립적인 표현 하나만 사용하는 대신, 입력의 각 위치는 관련된 다른 위치의 벡터를 가중합해 새로운 문맥 표현을 만듭니다. 이것이 시리즈의 큰 그림입니다. 다만 어텐션 가중치는 계산에 사용된 비율이며, 그것만으로 모델의 추론 이유나 사실성을 완전히 설명할 수는 없습니다.

## 먼저 전체 과정을 한눈에

1. **토큰화:** 텍스트를 토큰 ID로 변환합니다. 예시의 단어 단위 분할은 실제 토크나이저를 단순화한 것입니다.
2. **임베딩과 위치 정보:** 각 토큰 ID와 위치를 벡터로 표현합니다.
3. **어텐션:** Q·K로 점수를 구하고 소프트맥스로 비율을 만든 뒤 V를 가중합합니다.
4. **나머지 블록:** 잔차 연결, 정규화와 피드포워드 신경망 등을 거칩니다.
5. **출력:** 마지막 표현으로 다음 토큰의 점수와 확률을 계산합니다.

실제 모델은 이런 블록을 여러 층 쌓습니다. 트랜스포머는 2017년 기계 번역을 위한 인코더–디코더로 처음 제안되었고, GPT 계열은 주로 디코더 형태와 인과적 셀프 어텐션을 활용합니다. 어텐션은 이 구조의 핵심 구성 요소지만 모델 전체와 같은 말은 아닙니다.

## 완전한 코드: 점수를 문맥 벡터로

이번에는 Q·K를 아직 계산하지 않고 **점수를 직접 지정**합니다. 세 값은 한 위치가 세 토큰을 참고할 때 사용하는 가상의 점수입니다. 이 예제는 학습된 언어 모델이 아닙니다.

```python
import numpy as np

np.set_printoptions(precision=3, suppress=True)
tokens = ["나는", "사과를", "먹는다"]
values = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
scores = np.array([0.2, 1.2, 0.4])
exp_scores = np.exp(scores - np.max(scores))
weights = exp_scores / exp_scores.sum()
context = weights @ values

for token, weight in zip(tokens, weights):
    print(f"{token}: {weight:.3f}")
print("가중치 합:", round(weights.sum(), 3))
print("문맥 벡터:", context)
```

### 실행 결과

```text
나는: 0.202
사과를: 0.550
먹는다: 0.247
가중치 합: 1.0
문맥 벡터: [0.45  0.798]
```

## 한 줄씩 따라가기

- `import numpy as np`: 숫자 배열과 행렬 연산을 위한 NumPy를 불러옵니다.
- `set_printoptions(...)`: 출력 숫자를 소수점 세 자리로 보기 좋게 설정합니다. 계산 정밀도를 바꾸지는 않습니다.
- `tokens = ...`: 결과를 읽기 위한 이름입니다. 모델의 실제 토큰화 과정은 아닙니다.
- `values = ...`: 세 토큰의 2차원 값 벡터입니다. 모양은 `(토큰 수 3, 벡터 차원 2)`입니다. 계산을 위해 임의로 정했습니다.
- `scores = ...`: 한 위치에서 세 토큰에 매긴 점수입니다. 점수는 확률이 아니며 음수일 수도 있습니다.
- `scores - np.max(scores)`: 큰 수의 지수 연산이 넘치지 않도록 같은 최댓값을 빼 줍니다. 소프트맥스 결과의 비율은 바뀌지 않습니다.
- `np.exp(...)`: 점수를 양수로 바꿉니다.
- `exp_scores / exp_scores.sum()`: 합이 1이 되는 어텐션 가중치를 계산합니다.
- `weights @ values`: 각 값 벡터를 가중치만큼 곱해 더합니다. 결과는 길이 2인 문맥 벡터입니다.
- `for ...`와 `print(...)`: 각 가중치, 합, 결과 벡터를 확인합니다.

## 결과 읽기

‘사과를’의 점수가 가장 커서 가중치도 가장 큽니다. 결과 벡터의 첫 좌표는 대략 `0.202×1 + 0.550×0 + 0.247×1 = 0.449`입니다. 소수점 표시 때문에 프로그램 출력과 마지막 자리가 달라 보일 수 있습니다. 두 번째 좌표도 같은 방식으로 계산합니다.

**직접 바꿔 보기:** `scores`에서 첫 번째 값을 `3.0`으로 바꾸면 첫 번째 값 벡터의 영향이 커집니다. 점수 세 개가 같으면 가중치도 같아집니다.

## 스스로 확인하기

1. 점수가 아니라 가중치의 합이 1인 이유는 무엇일까요?
2. `values`의 벡터 차원을 2에서 4로 바꾸면 결과 차원은 무엇에 따라 결정될까요?
3. 가중치가 크다는 것과 모델의 설명이 완성되었다는 것은 왜 다를까요?

다음 글에서는 직접 정했던 점수를 Q와 K로 계산합니다.

## 참고 자료

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 트랜스포머 원 논문
- PyTorch, [Scaled Dot Product Attention 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — 스케일링과 마스킹을 포함한 구현 참고
- David Louapre, [Natural Language Processing before LLMs](https://huggingface.co/spaces/dlouapre/nlp-before-llms) — 어텐션까지 이어지는 자연어 처리의 흐름
