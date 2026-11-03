---
layout: '../../layouts/Post.astro'
title: '어텐션과 트랜스포머 4: 멀티헤드 어텐션'
description: '어텐션과 트랜스포머의 계산 원리를 PyTorch 학습 실습으로 이해합니다.'
date: '2026-11-03'
tags: ['attention', 'transformer', 'pytorch', 'deep-learning']
---

> **어텐션과 트랜스포머 · 4/5**

## 목차

- [실습 문제](#section-1)
- [여러 헤드를 사용하는 이유](#section-2)
- [헤드별 텐서 모양](#section-3)
- [전체 코드](#section-4)
- [실습 준비](#section-5)
- [결과를 비교하는 방법](#section-6)
- [직접 확인해 보기](#section-7)

멀티헤드 어텐션은 여러 개의 어텐션 연산을 병렬로 수행한 뒤 결과를 결합합니다. 각 헤드는 별도의 투영을 사용하므로 서로 다른 가중치를 학습할 수 있습니다. 이번 글에서는 두 항목의 서로 다른 속성을 예측하는 문제로 멀티헤드 어텐션을 학습하고, 헤드 수에 따른 결과를 비교합니다.

<a id="section-1"></a>
## 실습 문제

각 항목에는 이름, 색, 모양이 있습니다. 질문은 이름 두 개로 구성되며, 모델은 첫 번째 이름의 색과 두 번째 이름의 모양을 출력합니다.

```text
항목: A=(색 2, 모양 0), B=(색 1, 모양 3), …
질문: A, B
정답: 색 2, 모양 3
```

한 표본에는 서로 다른 이름 다섯 개가 있고 색과 모양은 각각 네 종류입니다. 질문의 두 이름은 같을 수도 있습니다. 이름과 속성의 조합은 표본마다 바뀌므로 모델은 현재 입력에 있는 값을 이용해야 합니다.

이 예제에서 Query는 두 질문 이름을 결합한 표현이고 Key는 각 항목의 이름 표현입니다. Value는 각 항목의 색과 모양을 결합한 표현입니다. 질문과 항목이 별도로 주어지므로 교차 어텐션을 사용합니다.

<a id="section-2"></a>
## 여러 헤드를 사용하는 이유

단일 헤드는 Query 하나에 대해 위치별 가중치 하나를 계산합니다. 이 가중치는 해당 헤드의 모든 Value 차원에 같이 적용됩니다. 서로 다른 위치의 정보를 함께 사용해야 하는 문제라면 여러 가중치 분포를 계산하고 결합하는 방식이 유용할 수 있습니다.

```text
headₕ = softmax(QₕKₕᵀ / √dₕ) Vₕ
output = concat(head₁, …, headH) Wₒ
```

헤드마다 사용하는 Q·K·V의 투영은 서로 다릅니다. 다만 ‘첫 번째 헤드는 색, 두 번째 헤드는 모양’을 반드시 담당한다고 가정해서는 안 됩니다. 모델은 출력 정답을 맞히도록 학습하며 헤드별 역할은 데이터와 파라미터에 따라 달라질 수 있습니다.

단일 헤드가 두 관계를 절대로 처리할 수 없다는 뜻도 아닙니다. 표현 차원, 입력 구성, 출력 계층과 학습 조건에 따라 다른 결과가 나올 수 있습니다. 이 점은 실제 비교 실험으로 확인해야 합니다.

<a id="section-3"></a>
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

<a id="section-4"></a>
## 전체 코드

<a id="section-5"></a>
## 실습 준비

Python 3.12에서 실행할 수 있습니다. PyTorch를 설치한 뒤 [week04.py](/attention-course/week04.py)를 내려받아 실행하세요. 아래 코드와 내려받는 파일은 같습니다.

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
    """두 카드 이름과 두 속성을 질의하는 검색 문제를 생성합니다.

    카드 값은 (배치, 카드 수, 2), 질문과 정답은 각각 (배치, 2)입니다.
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
        """마지막 표현 차원을 헤드별로 나눕니다.

        (B, T, D)를 (B, H, T, D/H)로 변환합니다.
        """
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.heads, self.dh).transpose(1, 2)

    def forward(self, keys, values, query, uniform=False):
        """두 질문을 헤드별로 처리해 색과 모양의 logits를 반환합니다.

        logits: (B, 2, 4), weights: (B, H, 1, 카드 수).
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
    """기울기를 기록하지 않고 손실과 정확도를 계산합니다.

    실습별로 선택 위치, 검색 위치 또는 답 위치도 함께 측정합니다.
    """
    model.eval()
    keys, values, query, target, _ = batch
    logits, _ = model(keys, values, query, uniform)
    correct = logits.argmax(-1) == target
    return {'loss': F.cross_entropy(logits.reshape(-1, 4), target.flatten()).item(),
            'accuracy': correct.float().mean().item(),
            'joint_accuracy': correct.all(-1).float().mean().item()}


def main():
    """실행 옵션을 읽고 학습, 검증과 결과 저장을 순서대로 수행합니다."""
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

<a id="section-6"></a>
## 결과를 비교하는 방법

기본 설정은 시드 44, 1200단계 학습입니다.

| 검증 지표 | 학습 전 | 마지막 단계 |
|---|---:|---:|
| 교차 엔트로피 | 1.41209 | 0.00002 |
| 정확도 | 22.27% | 100.00% |
| 공동 정확도 | 4.54% | 100.00% |

![검증 손실과 정확도 그래프](/attention-course/week04-learning.svg)

학습한 모델의 가중치를 균등하게 바꿨을 때 정확도는 48.85%였습니다.

같은 시드와 학습 단계에서 1헤드 모델의 공동 정확도는 46.48%, 2헤드 모델은 100.00%였습니다. 두 모델의 파라미터는 각각 7,080개, 7,080개입니다.

| 시드 | 마지막 단계의 검증 정확도 |
|---|---:|
| 44 | 100.00% |
| 54 | 100.00% |
| 64 | 100.00% |

[측정 기록 보기](/attention-course/week04-metrics.json)


`accuracy`는 두 정답을 각각 평가한 정확도입니다. `joint_accuracy`는 한 표본에서 두 정답을 모두 맞힌 비율입니다. 두 결과를 구분해야 합니다. 독립적인 균등 추측이라면 항목별 기대 정확도는 25%, 두 개를 모두 맞힐 기대 비율은 6.25%입니다.

1헤드와 2헤드 비교에서는 모델 차원 32를 유지합니다. 따라서 1헤드는 32차원, 2헤드는 각 16차원입니다. 파라미터 수와 학습 단계도 함께 확인할 수 있습니다. 한 시드의 결과만으로 모든 문제에서 특정 헤드 수가 더 좋다고 결론 내릴 수는 없습니다.

학습한 모델의 가중치를 균등하게 바꾸는 개입도 포함했습니다. 이는 헤드 수를 바꾼 모델을 처음부터 학습하는 실험과 다릅니다. 전자는 학습된 모델이 선택 가중치를 얼마나 사용하는지 확인하고, 후자는 서로 다른 구조의 학습 결과를 비교합니다.

<a id="section-7"></a>
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
