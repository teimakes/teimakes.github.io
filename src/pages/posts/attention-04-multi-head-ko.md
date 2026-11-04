---
layout: '../../layouts/Post.astro'
title: '어텐션과 트랜스포머 4: 멀티헤드 어텐션'
description: '멀티헤드 어텐션을 구현하고 두 질문을 동시에 푸는 과제에서 헤드별 가중치와 동일 파라미터 조건의 성능을 비교합니다.'
date: '2026-11-03'
tags: ['attention', 'transformer', 'pytorch', 'deep-learning']
---

> **어텐션과 트랜스포머 · 4/5**

멀티헤드 어텐션은 서로 다른 투영으로 여러 가중치 분포를 계산한 뒤 그 결과를 결합합니다. 각 헤드의 계산과 텐서 모양을 이해하려면 헤드를 나누는 시점, 정규화하는 축, 다시 결합하는 순서를 구분해야 합니다.

이 시리즈에서는 위치별 가중합, Q·K·V, 인과적 마스크를 거쳐 이번 글의 멀티헤드 연산을 구현합니다. 5편에서는 이를 디코더 블록에 넣고 다음 토큰 학습과 생성을 진행합니다. 이번 4편의 목표는 여러 헤드의 계산을 직접 구현하고 헤드 수를 바꾼 학습 결과를 비교하는 것입니다.

코드에는 서로 다른 두 항목의 속성을 함께 예측해야 하는 문제를 사용합니다. 전체 표현 차원과 파라미터 수는 유지하고 헤드 수만 바꿉니다. 각 답을 맞힌 비율과 두 답을 모두 맞힌 비율을 나누어 기록해, 어느 조건에서 차이가 생기는지 확인합니다.

## 목차

- [시리즈에서 다루는 내용](#section-1)
- [실습 문제](#section-2)
- [여러 헤드를 사용하는 이유](#section-3)
- [헤드별 텐서 모양](#section-4)
- [질문 두 개와 Query 길이 1](#section-5)
- [투영을 먼저 하고 헤드를 나누는 이유](#section-6)
- [파라미터 수를 맞춘 비교](#section-7)
- [헤드별 가중치를 분석할 때](#section-8)
- [전체 코드](#section-9)
- [실습 준비](#section-10)
- [결과를 비교하는 방법](#section-11)
- [직접 확인해 보기](#section-12)

<a id="section-1"></a>
## 시리즈에서 다루는 내용

| 글 | 확인할 내용 | 구현할 모델 |
|---|---|---|
| 1. 점수와 가중합 | 최종 정답으로 위치별 가중치를 학습하는 과정 | 표식 위치의 수치를 분류하는 모델 |
| 2. Q·K·V | 질문과 일치하는 항목을 참조하는 과정 | 이름에 연결된 값을 찾는 검색 모델 |
| 3. 인과적 셀프 어텐션 | 미래 입력을 참조하지 않도록 제한하는 방법 | 첫 숫자를 나중에 출력하는 복사 모델 |
| 4. 멀티헤드 | 여러 투영의 출력을 나누고 결합하는 방법 | 두 항목의 속성을 함께 찾는 모델 |
| 5. 디코더 언어 모델 | 다음 토큰 학습부터 저장과 생성까지의 실행 | 작은 기호 언어를 학습하는 모델 |

<a id="section-2"></a>
## 실습 문제

각 항목에는 이름, 색, 모양이 있습니다. 질문은 이름 두 개로 구성되며, 모델은 첫 번째 이름의 색과 두 번째 이름의 모양을 출력합니다.

```text
항목: A=(색 2, 모양 0), B=(색 1, 모양 3), …
질문: A, B
정답: 색 2, 모양 3
```

한 표본에는 서로 다른 이름 다섯 개가 있고 색과 모양은 각각 네 종류입니다. 질문의 두 이름은 같을 수도 있습니다. 이름과 속성의 조합은 표본마다 바뀌므로 모델은 현재 입력에 있는 값을 이용해야 합니다.

이 예제에서 Query는 두 질문 이름을 결합한 표현이고 Key는 각 항목의 이름 표현입니다. Value는 각 항목의 색과 모양을 결합한 표현입니다. 질문과 항목이 별도로 주어지므로 교차 어텐션을 사용합니다.

<a id="section-3"></a>
## 여러 헤드를 사용하는 이유

단일 헤드는 Query 하나에 대해 위치별 가중치 하나를 계산합니다. 이 가중치는 해당 헤드의 모든 Value 차원에 같이 적용됩니다. 서로 다른 위치의 정보를 함께 사용해야 하는 문제라면 여러 가중치 분포를 계산하고 결합하는 방식이 유용할 수 있습니다.

```text
headₕ = softmax(QₕKₕᵀ / √dₕ) Vₕ
output = concat(head₁, …, headH) Wₒ
```

헤드마다 사용하는 Q·K·V의 투영은 서로 다릅니다. 다만 ‘첫 번째 헤드는 색, 두 번째 헤드는 모양’을 반드시 담당한다고 가정해서는 안 됩니다. 모델은 출력 정답을 맞히도록 학습하며 헤드별 역할은 데이터와 파라미터에 따라 달라질 수 있습니다.

단일 헤드가 두 관계를 절대로 처리할 수 없다는 뜻도 아닙니다. 표현 차원, 입력 구성, 출력 계층과 학습 조건에 따라 다른 결과가 나올 수 있습니다. 이 점은 실제 비교 실험으로 확인해야 합니다.

<a id="section-4"></a>
## 헤드별 텐서 모양

모델 차원 D는 32, 헤드 수 H는 2입니다. 헤드별 차원은 `D/H = 16`입니다. 질문 길이는 1, 항목 수는 5입니다.

```text
투영된 Q: (B, 1, 32) → (B, 2, 1, 16)
투영된 K: (B, 5, 32) → (B, 2, 5, 16)
투영된 V: (B, 5, 32) → (B, 2, 5, 16)
점수:      (B, 2, 1, 5)
헤드 출력: (B, 2, 1, 16) → 결합 후 (B, 32)
```

`reshape()`는 각 표현 차원을 헤드 수와 헤드별 차원으로 나누고, `transpose()`는 헤드 축을 앞으로 옮깁니다. 점수 계산에서 마지막 두 축은 행렬곱에 사용되고 앞의 배치·헤드 축은 각각 독립적으로 유지됩니다. 소프트맥스는 마지막 항목 축에 적용합니다.

헤드 출력을 결합할 때는 축 순서를 되돌린 다음 `contiguous()`와 `reshape()`를 사용합니다. 결합된 결과에 출력 투영을 적용하고, 최종 분류기는 `(B, 2, 4)` 형태의 logits를 만듭니다. 두 번째 축은 두 정답을, 마지막 축은 각 정답의 네 클래스를 나타냅니다.

<a id="section-5"></a>
## 질문 두 개와 Query 길이 1

이 구현에서 혼동하기 쉬운 부분은 질문의 개수와 어텐션 Query 축의 길이입니다. `query`에는 이름 ID가 두 개 있지만, 두 임베딩을 이어 붙여 하나의 입력 벡터로 만듭니다. 따라서 투영한 Query의 길이 축은 1입니다.

각 헤드는 이 결합된 질문에 서로 다른 Wq를 적용한 표현을 사용합니다. 한 헤드를 첫 질문에, 다른 헤드를 두 번째 질문에 강제로 대응시키지는 않습니다. 두 답에 대한 분류 손실이 줄어드는 방향으로 전체 파라미터가 학습됩니다.

이 구성과 길이 2인 Query 시퀀스를 사용하는 구성은 다릅니다. 후자에서는 두 질문 위치마다 어텐션을 따로 계산합니다. 이 글의 목적은 한 Query가 여러 위치의 서로 다른 속성을 함께 사용해야 할 때 헤드 수에 따른 결과를 비교하는 것입니다.

<a id="section-6"></a>
## 투영을 먼저 하고 헤드를 나누는 이유

수식에서는 헤드마다 Wq, Wk, Wv를 따로 씁니다. 코드는 전체 차원 D로 한 번 투영한 뒤 마지막 차원을 헤드별로 나눕니다. 큰 투영 행렬의 출력 열을 헤드별로 나눈 것과 같은 계산입니다. 헤드를 나눈다는 이유만으로 입력의 고정된 일부 특성만 사용하는 것은 아닙니다. 투영 행렬이 입력 전체를 이용해 각 헤드의 표현을 만듭니다.

`split()`의 입력이 `(B, T, 32)`이고 헤드 수가 2이면 먼저 `(B, T, 2, 16)`으로 바꿉니다. 이후 `(B, 2, T, 16)`으로 전치합니다. 마지막 두 축으로 행렬곱을 하고, 앞의 배치와 헤드 축은 독립적으로 유지합니다.

최종 결합은 이 과정의 축 순서를 되돌립니다. `contexts`를 `(B, 1, H, Dh)`로 바꾸고 헤드 차원을 이어 붙여 `(B, D)`로 만든 뒤 Wo를 적용합니다. 두 헤드의 가중치를 평균내는 연산이 아니라 헤드별 문맥을 보존해 결합하는 연산입니다.

<a id="section-7"></a>
## 파라미터 수를 맞춘 비교

이 코드에서는 헤드 수만 바꾸고 D는 32로 고정합니다. 헤드별 차원은 바뀌지만 전체 투영 행렬 크기는 같습니다. 임베딩과 출력 분류기를 포함한 파라미터 수는 다음과 같습니다.

```text
임베딩: 이름 12D + 색 4D + 모양 4D = 20D
Wq·Wk·Wv·Wo: 2D² + D² + 2D² + D² = 6D²
출력 계층의 가중치: 8D
편향: Wo의 D + 분류기의 8
합계: 6D² + 29D + 8 = 7,080  (D=32)
```

헤드 수를 늘리면서 헤드별 차원까지 유지하면 D도 커지므로 다른 비교가 됩니다. 이 경우 성능 차이에 파라미터 증가 효과가 포함됩니다. 실험 목적에 따라 무엇을 고정했는지 먼저 정해야 합니다.

<a id="section-8"></a>
## 헤드별 가중치를 분석할 때

출력한 `sample_chosen`은 정답 두 개에 해당하는 항목 위치입니다. `sample_weights`는 각 헤드가 다섯 항목에 준 가중치입니다. 두 값을 함께 보면 헤드가 어느 위치를 주로 참조하는지 확인할 수 있습니다.

한 표본의 가중치만으로 헤드의 역할을 정하지는 마세요. 여러 표본에서 두 질문이 같은 경우와 다른 경우를 나눠 보고, 헤드 번호가 바뀌어도 예측 기능이 같을 수 있다는 점을 고려해야 합니다. 최종 분류기는 모든 헤드의 문맥을 이용하므로 헤드 하나의 최대 위치와 답 하나를 항상 동일시할 수는 없습니다.

<a id="section-9"></a>
## 전체 코드

<a id="section-10"></a>
## 실습 준비

Python 3.12에서 실행할 수 있습니다. PyTorch를 설치한 뒤 [week04.py](https://teimakes.github.io/attention-course/week04.py)를 내려받아 실행하세요. 아래 코드와 내려받는 파일은 같습니다.

```bash
python -m pip install torch==2.15.0
python week04.py
```

결과는 기본적으로 `runs/week04`에 저장됩니다. `--seed`, `--steps`, `--out` 옵션으로 시드, 학습 단계, 저장 경로를 지정할 수 있습니다.

```python
"""Week 4: multi-head cross-attention for two independent lookups."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def make_batch(size, generator):
    """한 이름의 색과 다른 이름의 모양을 동시에 질문합니다.

    keys: (B, 5). 표본마다 중복되지 않는 이름 다섯 개입니다.
    values: (B, 5, 2). 마지막 축은 색, 모양이며 각각 0~3입니다.
    chosen: (B, 2). 첫 질문과 두 번째 질문의 항목 위치입니다.
    query = keys.gather(1, chosen): (B, 2) 질문 이름을 구합니다.
    target의 첫 열은 첫 질문 항목의 색, 둘째 열은 두 번째 항목의 모양입니다.
    stack(..., dim=-1)으로 두 정답을 (B, 2)로 결합합니다.

    두 질문이 같은 항목을 가리킬 수도 있습니다. chosen은 학습 손실이
    아니라 가중치 분석에 사용하도록 정답과 함께 반환합니다.
    """
    keys = torch.rand(size, 12, generator=generator).argsort(-1)[:, :5]
    values = torch.randint(4, (size, 5, 2), generator=generator)
    chosen = torch.randint(5, (size, 2), generator=generator)
    rows = torch.arange(size)
    query = keys.gather(1, chosen)
    target = torch.stack([values[rows, chosen[:, 0], 0],
                          values[rows, chosen[:, 1], 1]], dim=-1)
    return keys, values, query, target, chosen


class MultiRetriever(nn.Module):
    def __init__(self, heads=2, dim=32):
        """같은 총 차원에서 헤드 수를 바꿀 수 있는 검색 모델을 만듭니다.

        dim은 heads로 나누어져야 하며 dh=dim/heads입니다.
        key는 이름 12종, color와 shape은 각 속성 4종의 임베딩입니다.
        wq는 두 질문 임베딩을 합친 2*dim차원을 dim으로 변환합니다.
        wk는 이름 임베딩, wv는 두 속성의 결합을 각각 dim으로 변환합니다.
        wo는 결합된 헤드 출력을 투영합니다.
        classifier는 색 4개와 모양 4개의 점수를 합쳐 8개 출력합니다.

        heads만 바꾸면 각 헤드의 차원은 달라지지만 전체 파라미터 수는 같습니다.
        """
        super().__init__()
        if dim % heads:
            raise ValueError('dim must be divisible by heads')
        self.heads, self.dim, self.dh = heads, dim, dim // heads
        self.key = nn.Embedding(12, dim)
        self.color = nn.Embedding(4, dim)
        self.shape = nn.Embedding(4, dim)
        self.wq = nn.Linear(2 * dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(2 * dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim)
        self.classifier = nn.Linear(dim, 8)

    def split(self, x):
        """전체 표현을 헤드별 표현으로 나누고 축 순서를 바꿉니다.

        x: (B, T, D).
        reshape(B, T, H, Dh): 마지막 D차원을 H*Dh로 나눕니다.
        transpose(1, 2): 결과를 (B, H, T, Dh) 순서로 바꿉니다.
        이 배치에서는 헤드별 행렬곱을 서로 독립적으로 계산할 수 있습니다.
        """
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.heads, self.dh).transpose(1, 2)

    def forward(self, keys, values, query, uniform=False):
        """두 질문에서 만든 Query로 항목의 속성을 검색합니다.

        keys: (B, 5), values: (B, 5, 2), query: (B, 2)입니다.
        q_input: 두 이름 임베딩을 순서대로 이어 (B, 1, 2D)로 만듭니다.
            질문이 두 개여도 이 구현의 Query 길이는 1입니다.
        value_input: 색·모양 임베딩을 합쳐 (B, 5, 2D)로 만듭니다.
        q/k/v: 투영과 split 후 (B,H,1,Dh), (B,H,5,Dh), (B,H,5,Dh)입니다.
        scores: Q·K 내적을 sqrt(Dh)로 나눈 (B,H,1,5) 텐서입니다.
        weights: 다섯 항목 축을 정규화합니다. 헤드 축은 정규화하지 않습니다.
        contexts: Value 가중합이며 (B,H,1,Dh)입니다.
        merged: 축을 되돌려 헤드들을 이어 붙인 (B,D) 문맥입니다.
        wo와 classifier: 이를 색과 모양의 (B,2,4) logits로 바꿉니다.

        반환한 weights에서 각 헤드가 두 질문 중 어느 항목에 가중치를
        주는지 확인할 수 있습니다. 헤드 역할은 미리 지정하지 않습니다.
        """
        batch = keys.size(0)
        q_input = self.key(query).reshape(batch, 1, 2 * self.dim)
        value_input = torch.cat([self.color(values[..., 0]),
                                 self.shape(values[..., 1])], dim=-1)
        q = self.split(self.wq(q_input))
        k = self.split(self.wk(self.key(keys)))
        v = self.split(self.wv(value_input))
        # 각 query와 key의 내적으로 위치별 유사도를 계산합니다.
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        # 위치별 점수를 확률로 바꾸어 어느 입력을 참고할지 정합니다.
        weights = scores.softmax(-1)
        if uniform:
            weights = torch.ones_like(weights) / keys.size(1)
        contexts = weights @ v
        # 헤드별 문맥을 위치별로 다시 이어 붙입니다.
        merged = contexts.transpose(1, 2).contiguous().reshape(batch, self.dim)
        # 두 질문의 정답을 각각 네 개의 클래스 중에서 예측합니다.
        logits = self.classifier(self.wo(merged)).reshape(batch, 2, 4)
        return logits, weights


@torch.no_grad()
def evaluate(model, batch, uniform=False):
    """항목별 정확도와 두 답을 모두 맞힌 비율을 측정합니다.

    logits.reshape(-1, 4)는 (B,2,4)를 (B*2,4)로 변환합니다.
    target.flatten()은 같은 순서의 (B*2,) 라벨입니다.
    cross_entropy는 두 답에 대한 평균 손실입니다.
    correct는 (B,2)이며 각 답의 정답 여부를 표시합니다.
    accuracy는 모든 답의 평균 정확도입니다.
    joint_accuracy는 correct.all(-1), 즉 두 답이 모두 맞은 표본의 비율입니다.

    uniform을 지정하면 모든 헤드에서 항목 가중치를 동일하게 바꾸어 평가합니다.
    """
    model.eval()
    keys, values, query, target, _ = batch
    logits, _ = model(keys, values, query, uniform)
    correct = logits.argmax(-1) == target
    return {'loss': F.cross_entropy(logits.reshape(-1, 4), target.flatten()).item(),
            'accuracy': correct.float().mean().item(),
            'joint_accuracy': correct.all(-1).float().mean().item()}


def main():
    """헤드 수에 따른 두 속성 검색 모델의 학습 결과를 기록합니다.

    --heads는 총 차원 32를 나눌 헤드 수입니다. 다른 조건을 유지한
    헤드 수 비교가 가능하도록 임베딩과 전체 투영 크기는 바꾸지 않습니다.
    검증 2,048개를 고정하고 매 단계 학습 표본 128개를 새로 생성합니다.
    두 답의 logits를 (B*2,4)로, target을 (B*2,)로 바꿔 손실을 계산합니다.
    역전파와 AdamW 갱신을 수행하고 200단계마다 항목별·공동 정확도를 기록합니다.

    마지막에는 가중치의 (1,H,1,5) 모양과 항목 축 합을 검사합니다.
    파라미터 수, 헤드별 표본 가중치, 균등 개입 결과를 metrics.json에
    함께 저장하고 model.pt에 모델 상태를 보관합니다.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--out', default='runs/week04')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    train_rng = torch.Generator().manual_seed(args.seed + 1)
    validation = make_batch(2048, torch.Generator().manual_seed(args.seed + 1000))
    model = MultiRetriever(args.heads)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    history = [{'step': 0, **evaluate(model, validation)}]
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        keys, values, query, target, _ = make_batch(128, train_rng)
        logits, _ = model(keys, values, query)
        # 정답과 logits를 비교합니다. 소프트맥스는 손실 함수 내부에서 처리합니다.
        loss = F.cross_entropy(logits.reshape(-1, 4), target.flatten())
        # 이전 배치의 기울기를 비우고 현재 손실을 역전파합니다.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # 기울기가 지나치게 커질 때 업데이트의 크기를 제한합니다.
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 계산된 기울기로 학습 가능한 파라미터를 갱신합니다.
        optimizer.step()
        if step % 200 == 0 or step == args.steps:
            metrics = {'step': step, 'train_loss': loss.item(), **evaluate(model, validation)}
            history.append(metrics)
            print(json.dumps(metrics))
    model.eval()
    with torch.no_grad():
        keys, values, query, _, chosen = validation
        _, weights = model(keys[:1], values[:1], query[:1])
        assert weights.shape == (1, args.heads, 1, 5)
        assert torch.allclose(weights.sum(-1), torch.ones(1, args.heads, 1), atol=1e-6)
    report = {'torch': torch.__version__, 'seed': args.seed, 'heads': args.heads,
              'parameters': sum(p.numel() for p in model.parameters()),
              'seconds': time.perf_counter() - started, 'history': history,
              'uniform_intervention': evaluate(model, validation, True),
              'sample_chosen': chosen[0].tolist(), 'sample_weights': weights[0, :, 0].tolist()}
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    torch.save(model.state_dict(), output / 'model.pt')
    print(json.dumps({k: v for k, v in report.items() if k != 'history'}, indent=2))


if __name__ == '__main__':
    main()
```

<a id="section-11"></a>
## 결과를 비교하는 방법

기본 설정은 시드 44, 1200단계 학습입니다.

| 검증 지표 | 학습 전 | 마지막 단계 |
|---|---:|---:|
| 교차 엔트로피 | 1.41209 | 0.00002 |
| 정확도 | 22.27% | 100.00% |
| 공동 정확도 | 4.54% | 100.00% |

![검증 손실과 정확도 그래프](https://teimakes.github.io/attention-course/week04-learning.svg)

학습한 모델의 가중치를 균등하게 바꿨을 때 정확도는 48.85%였습니다.

같은 시드와 학습 단계에서 1헤드 모델의 공동 정확도는 46.48%, 2헤드 모델은 100.00%였습니다. 두 모델의 파라미터는 각각 7,080개, 7,080개입니다.

| 시드 | 마지막 단계의 검증 정확도 |
|---|---:|
| 44 | 100.00% |
| 54 | 100.00% |
| 64 | 100.00% |

[측정 기록 보기](https://teimakes.github.io/attention-course/week04-metrics.json)


`accuracy`는 두 정답을 각각 평가한 정확도입니다. `joint_accuracy`는 한 표본에서 두 정답을 모두 맞힌 비율입니다. 두 결과를 구분해야 합니다. 독립적인 균등 추측이라면 항목별 기대 정확도는 25%, 두 개를 모두 맞힐 기대 비율은 6.25%입니다.

1헤드와 2헤드 비교에서는 모델 차원 32를 유지합니다. 따라서 1헤드는 32차원, 2헤드는 각 16차원입니다. 파라미터 수와 학습 단계도 함께 확인할 수 있습니다. 한 시드의 결과만으로 모든 문제에서 특정 헤드 수가 더 좋다고 결론 내릴 수는 없습니다.

학습한 모델의 가중치를 균등하게 바꾸는 개입도 포함했습니다. 이는 헤드 수를 바꾼 모델을 처음부터 학습하는 실험과 다릅니다. 전자는 학습된 모델이 선택 가중치를 얼마나 사용하는지 확인하고, 후자는 서로 다른 구조의 학습 결과를 비교합니다.

<a id="section-12"></a>
## 직접 확인해 보기

1. `--heads 1`, `--heads 2`, `--heads 4`를 각각 세 시드로 실행합니다. 공동 정확도와 파라미터 수를 함께 기록합니다.
2. 두 질문의 이름이 같은 경우와 다른 경우를 나누어 평가합니다.
3. 각 헤드에서 가중치가 가장 높은 위치와 두 정답 위치를 비교합니다. 헤드 번호에 정해진 역할이 있다고 가정하지 말고 여러 표본을 확인합니다.

다음 글에서는 멀티헤드 셀프 어텐션을 포함한 디코더 언어 모델을 학습하고 생성까지 진행합니다.

## 함께 읽을 글

1. [어텐션의 점수와 가중합](/posts/attention-01-big-picture-ko/)
2. [Q·K·V와 어텐션 학습](/posts/attention-02-qkv-ko/)
3. [인과적 셀프 어텐션](/posts/attention-03-causal-mask-ko/)
4. [멀티헤드 어텐션](/posts/attention-04-multi-head-ko/)
5. [트랜스포머 언어 모델 학습과 생성](/posts/attention-05-transformer-block-ko/)

## 참고 자료

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html)
