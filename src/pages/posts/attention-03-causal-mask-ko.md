---
layout: '../../layouts/Post.astro'
title: '어텐션 이해하기 ③: GPT는 왜 미래 토큰을 가릴까?'
description: '인과적 마스크를 직접 구현하고 미래 토큰이 앞쪽 출력에 영향을 주지 않는지 검사합니다.'
date: '2026-11-03'
tags: ['attention', 'transformer', 'llm', 'python']
---

> **어텐션과 트랜스포머 시리즈 · 3/5**  
> 큰 그림부터 실행 코드까지 차근차근 살펴봅니다. 각 글의 코드는 독립 실행 가능합니다.

## 시리즈 목차

1. [어텐션 이해하기 ①: 큰 그림부터 가중합까지](/posts/attention-01-big-picture-ko/)
2. [어텐션 이해하기 ②: Q·K·V와 스케일드 닷 프로덕트](/posts/attention-02-qkv-ko/)
3. [어텐션 이해하기 ③: GPT는 왜 미래 토큰을 가릴까?](/posts/attention-03-causal-mask-ko/)
4. [어텐션 이해하기 ④: 멀티헤드와 텐서 모양 추적하기](/posts/attention-04-multi-head-ko/)
5. [어텐션 이해하기 ⑤: 작은 트랜스포머 디코더 직접 만들기](/posts/attention-05-transformer-block-ko/)

**실행 환경:** Python 3와 NumPy가 필요합니다. 터미널에서 `python -m pip install numpy`를 실행한 뒤, 아래 코드를 하나의 `.py` 파일로 저장해 실행하거나 노트북 셀에 붙여 넣으세요.

## 다음 토큰 예측과 정보 누출

GPT 방식의 언어 모델은 현재 위치까지의 토큰을 사용해 다음 토큰을 예측합니다. 학습 중 모든 위치를 한꺼번에 계산하더라도, 각 위치가 그 뒤에 있는 정답 토큰을 미리 보면 안 됩니다. 이를 방지하는 것이 **인과적 마스크(causal mask)**입니다.

길이 3의 입력에서 행은 query 위치, 열은 key 위치라고 하면 접근 가능 범위는 다음과 같습니다.

- 첫 번째 행: 첫 토큰만 볼 수 있습니다.
- 두 번째 행: 첫 번째와 두 번째 토큰을 볼 수 있습니다.
- 세 번째 행: 세 위치를 모두 볼 수 있습니다.

여기서 각 행은 **자기 자신을 포함한 이전 위치**를 참고합니다. 학습 시 위치 `t`의 출력은 보통 그 다음 위치 `t+1`의 토큰을 맞히는 데 사용합니다. 입력과 정답을 한 칸 어긋나게 구성해야 한다는 점이 중요합니다.

## 완전한 코드: 마스크와 미래 변경 실험

Q·K·V를 모두 입력과 같게 둔 단순화된 예제입니다. 학습 가능한 투영은 ②편, 여러 헤드는 ④편에서 다룹니다.

```python
import numpy as np

np.set_printoptions(precision=3, suppress=True)

def causal_attention(X):
    Q = K = V = X
    scores = Q @ K.T / np.sqrt(K.shape[-1])
    length = X.shape[0]
    future = np.triu(np.ones((length, length), dtype=bool), k=1)
    scores = np.where(future, -np.inf, scores)
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    return weights @ V, weights

X = np.array([[1., 0.], [0., 1.], [1., 1.]])
output, weights = causal_attention(X)
changed = X.copy()
changed[2] = [9., -4.]
changed_output, _ = causal_attention(changed)

print("가중치:\n", weights)
print("출력:\n", output)
print("미래 변경 후 앞 두 출력 동일:", np.allclose(output[:2], changed_output[:2]))
assert np.all(weights[np.triu_indices(3, k=1)] == 0)
assert np.allclose(output[:2], changed_output[:2])
```

### 실행 결과

```text
가중치:
 [[1.    0.    0.   ]
 [0.33  0.67  0.   ]
 [0.248 0.248 0.503]]
출력:
 [[1.    0.   ]
 [0.33  0.67 ]
 [0.752 0.752]]
미래 변경 후 앞 두 출력 동일: True
```

## 한 줄씩 따라가기

- `causal_attention(X)`는 입력 행렬을 받아 문맥 벡터와 가중치를 반환합니다.
- `Q = K = V = X`는 세 투영을 생략하기 위한 설정입니다. 실제 모델이 항상 동일하게 둔다는 뜻은 아닙니다.
- `scores = ...`는 각 query와 key의 내적을 구하고 차원의 제곱근으로 나눕니다.
- `length = X.shape[0]`은 시퀀스의 토큰 수를 구합니다.
- `np.ones(..., dtype=bool)`은 참으로 채운 정사각 행렬을 만듭니다.
- `np.triu(..., k=1)`은 대각선보다 위쪽, 즉 각 행의 미래 열만 참으로 남깁니다. 대각선은 허용됩니다.
- `np.where(future, -np.inf, scores)`는 금지된 점수를 음의 무한대로 바꿉니다. 마스크는 반드시 소프트맥스 전에 적용합니다.
- `shifted`, `exp_scores`, `weights`는 안정적인 행별 소프트맥스를 계산합니다. `exp(-inf)`는 0이므로 미래 위치의 가중치는 0입니다. 이 예제는 각 행에서 최소 한 위치가 허용되므로 유효한 분모가 생깁니다.
- `weights @ V`는 허용된 위치만 가중합합니다.
- `changed[2] = ...`는 세 번째, 즉 앞의 두 위치에서 미래에 해당하는 입력만 바꿉니다.
- `np.allclose(...)`는 미래 입력을 바꾸기 전과 후의 앞 두 출력을 비교합니다.
- 마지막 두 `assert`는 미래 가중치가 0이고 앞 두 출력이 같다는 성질을 확인합니다.

## 결과 읽기

가중치 행렬의 대각선 위가 모두 0인지 확인하세요. 세 번째 입력을 크게 바꿔도 첫 번째와 두 번째 출력은 같습니다. 세 번째 출력은 자기 자신의 입력이 바뀌었으므로 달라질 수 있습니다.

**직접 바꿔 보기:** 마스크 줄을 제거하고 미래 변경 실험을 다시 실행해 보세요. 앞쪽 출력도 달라질 수 있습니다. 이는 계산 과정을 관찰하는 실험이지 실제 모델이 정답을 이해했음을 뜻하지 않습니다.

## 주의할 점

학습할 때는 마스킹된 행렬 연산으로 여러 위치를 병렬 계산할 수 있지만, 새로운 토큰을 생성할 때는 앞서 생성한 결과에 의존합니다. 구현에 따라 KV 캐시로 과거 계산을 재사용할 수 있습니다. 패딩 마스크는 인과 마스크와 목적이 다르며 실제 배치에서는 둘을 함께 고려하기도 합니다.

다음 글에서는 여러 개의 어텐션 헤드를 병렬로 계산합니다.

## 참고 자료

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 트랜스포머 원 논문
- PyTorch, [Scaled Dot Product Attention 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — 스케일링과 마스킹을 포함한 구현 참고
- David Louapre, [Natural Language Processing before LLMs](https://huggingface.co/spaces/dlouapre/nlp-before-llms) — 어텐션까지 이어지는 자연어 처리의 흐름
