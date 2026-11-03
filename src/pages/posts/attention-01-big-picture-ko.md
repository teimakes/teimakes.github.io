---
layout: '../../layouts/Post.astro'
title: '어텐션과 트랜스포머 1: 어텐션의 점수와 가중합'
description: '어텐션과 트랜스포머의 계산 원리를 PyTorch 학습 실습으로 이해합니다.'
date: '2026-11-03'
tags: ['attention', 'transformer', 'pytorch', 'deep-learning']
---

> **어텐션과 트랜스포머 · 1/5**

## 목차

- [실습 문제](#section-1)
- [점수와 가중합](#section-2)
- [분류 정답만으로 가중치를 학습하는 과정](#section-3)
- [코드에서 확인할 부분](#section-4)
- [실습 준비](#section-5)
- [학습 결과](#section-6)
- [직접 확인해 보기](#section-7)

어텐션은 여러 입력 중 어떤 정보를 얼마나 반영할지 계산하는 방법입니다. 언어 모델은 이 계산을 이용해 각 토큰의 표현에 다른 위치의 정보를 반영합니다. 이번 글에서는 어텐션의 가장 기본이 되는 **점수 계산, 소프트맥스, 가중합**을 구현하고, 이 계산에 쓰이는 파라미터가 학습 과정에서 어떻게 바뀌는지 확인합니다.

트랜스포머는 토큰 임베딩에 위치 정보를 더한 뒤 여러 블록을 통과시킵니다. 각 블록에는 어텐션과 피드포워드 신경망이 들어갑니다. 언어 모델은 마지막으로 각 위치의 표현을 어휘 크기의 점수로 변환하고 다음 토큰을 예측합니다. 아래 그림은 5편에서 구현할 디코더의 구조입니다. 이번 글에서는 이 중 어텐션의 가중합을 먼저 다룹니다.

![토큰 임베딩, 어텐션, FFN과 출력 계층으로 구성된 디코더 구조](https://teimakes.github.io/attention-course/architecture.svg)

<a id="section-1"></a>
## 실습 문제

입력은 여덟 개의 위치로 구성됩니다. 각 위치에는 표식과 수치가 하나씩 있습니다. 표식이 1인 위치는 한 곳뿐이며, 정답은 그 위치의 수치가 양수인지 여부입니다.

```text
입력: [표식 0, 수치 -0.3], [표식 1, 수치 0.8], [표식 0, 수치 -1.4], …
정답: 1  (표식이 있는 위치의 수치가 양수)
```

표식의 위치와 수치는 매번 무작위로 정합니다. 모델은 전체 입력을 받고 0 또는 1을 예측합니다. 이때 어느 위치가 표식인지 알려주는 별도의 손실은 사용하지 않습니다. 최종 분류 정답만으로 학습한 뒤, 모델이 표식 위치에 어느 정도의 가중치를 주는지 확인합니다.

실제 언어 데이터에는 표식처럼 정답 위치를 명시해 주는 특성이 없습니다. 이 실습은 어텐션의 계산과 학습을 확인하기 위해 조건을 단순하게 정한 문제입니다.

<a id="section-2"></a>
## 점수와 가중합

입력의 i번째 위치를 xᵢ라고 합시다. 점수 함수는 각 입력을 실수 sᵢ로 바꿉니다. 이 값은 아직 확률이 아닙니다. 소프트맥스를 적용하면 위치별 가중치 αᵢ를 얻습니다.

```text
sᵢ = w₂ᵀ tanh(W₁xᵢ + b₁) + b₂
αᵢ = exp(sᵢ) / Σⱼ exp(sⱼ)
vᵢ = Wᵥxᵢ + bᵥ
c  = Σᵢ αᵢvᵢ
```

가중치 αᵢ는 0 이상이며 모든 위치의 합은 1입니다. 입력 하나당 가중치 하나를 계산하고, 그 가중치를 같은 위치의 값 벡터 vᵢ에 곱한 뒤 더하면 문맥 벡터 c가 됩니다. 모델은 c를 분류 계층에 넣어 두 클래스의 점수를 계산합니다.

소프트맥스는 각 표본의 위치 축에 적용해야 합니다. 배치에 속한 서로 다른 표본을 함께 정규화하면 한 표본의 입력이 다른 표본의 가중치에 영향을 줍니다. 코드의 `softmax(dim=-1)`은 `(배치, 위치)` 형태인 점수에서 마지막 축을 정규화합니다.

### 숫자로 계산해 보기

세 위치의 점수가 `[0, 1, 2]`라면 소프트맥스 가중치는 약 `[0.090, 0.245, 0.665]`입니다. 각 위치의 Value가 1, 2, 3이면 가중합은 다음과 같습니다.

```text
0.090 × 1 + 0.245 × 2 + 0.665 × 3 ≈ 2.575
```

모든 점수에 같은 수를 더해도 가중치는 같습니다. 소프트맥스는 점수의 상대적인 차이로 분포를 계산합니다. 구현할 때 최댓값을 먼저 빼면 지수 연산에서 숫자가 지나치게 커지는 문제를 줄일 수 있습니다. PyTorch의 `softmax()`는 이 계산을 처리합니다.

<a id="section-3"></a>
## 분류 정답만으로 가중치를 학습하는 과정

분류 결과가 정답과 다르면 교차 엔트로피 손실이 커집니다. `loss.backward()`는 손실에 대한 기울기를 분류 계층, 문맥 벡터, 가중치, 점수 함수 순서로 계산합니다. `optimizer.step()`은 이 기울기를 이용해 파라미터를 갱신합니다.

문맥 벡터에 대한 손실의 기울기를 g라고 하면, 위치 i의 점수에 대한 기울기는 다음과 같습니다.

```text
g = ∂L/∂c
∂L/∂sᵢ = αᵢ · gᵀ(vᵢ − c)
```

위 식에서 점수의 변화는 해당 위치의 값 벡터와 현재 문맥 벡터의 차이, 그리고 그 차이가 손실에 미치는 영향에 따라 결정됩니다. 이 계산이 반복되면서 점수 함수와 값 변환 계층이 함께 학습됩니다.

소프트맥스의 미분식 `∂αᵢ/∂sⱼ = αᵢ(δᵢⱼ − αⱼ)`도 확인해 보세요. δᵢⱼ는 두 인덱스가 같으면 1이고 다르면 0입니다. 한 위치의 점수를 바꾸면 정규화에 참여하는 다른 위치의 가중치도 함께 바뀝니다.

<a id="section-4"></a>
## 코드에서 확인할 부분

`AttentionClassifier`는 점수 함수 `scorer`, 값 변환 `value`, 분류기 `classifier`로 구성됩니다. 입력 크기가 `(B, T, 2)`일 때 점수와 가중치는 `(B, T)`, 값 벡터는 `(B, T, 16)`입니다. 가중합 결과는 `(B, 16)`, 분류 logits는 `(B, 2)`입니다. 여기서 B는 배치 크기, T는 입력 위치 수입니다.

`make_batch()`는 입력과 정답을 생성합니다. 반환값에 선택 위치가 포함되지만, 학습 손실은 `labels`만 사용합니다. 선택 위치는 평가할 때 해당 위치의 평균 가중치를 구하는 데 사용합니다. 학습 데이터와 검증 데이터는 서로 다른 난수 생성기로 만듭니다.

### 학습에 사용되는 정보

`values`는 정규분포에서 뽑은 수치이고 `selected`는 정답을 결정할 위치입니다. `one_hot()`으로 만든 표식은 입력에 포함됩니다. 모델은 표식으로 위치를 구별할 수 있지만 정답 수치의 부호는 직접 계산해야 합니다. `labels`는 선택된 수치가 0보다 큰지에 따라 0 또는 1로 정합니다.

`model.train()`으로 학습 모드를 설정하고 새 배치를 생성합니다. `zero_grad()`는 이전 단계의 기울기를 비우고 `backward()`는 현재 손실에 대한 기울기를 계산합니다. `clip_grad_norm_()`은 기울기 노름을 제한하고 AdamW의 `step()`이 파라미터를 갱신합니다.

평가는 `model.eval()`과 `torch.no_grad()`를 함께 사용합니다. 전자는 평가 모드를 설정하고 후자는 기울기 기록을 중지합니다. 역할이 다르므로 하나가 다른 하나를 대신하지 않습니다. 검증 표본은 고정해 두어 학습 전후를 같은 데이터에서 비교합니다.

<a id="section-5"></a>
## 실습 준비

Python 3.12에서 실행할 수 있습니다. PyTorch를 설치한 뒤 [week01.py](https://teimakes.github.io/attention-course/week01.py)를 내려받아 실행하세요. 아래 코드와 내려받는 파일은 같습니다.

```bash
python -m pip install torch==2.15.0
python week01.py
```

결과는 기본적으로 `runs/week01`에 저장됩니다. `--seed`, `--steps`, `--out` 옵션으로 시드, 학습 단계, 저장 경로를 지정할 수 있습니다.

```python
"""Week 1: learn attention pooling from classification supervision."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def make_batch(size, generator, length=8):
    """표식과 수치로 구성된 입력을 만들고, 표식 위치의 수치 부호를 정답으로 반환합니다.

    입력은 (배치, 위치, 2), 정답과 선택 위치는 각각 (배치,)입니다.
    """
    values = torch.randn(size, length, generator=generator)
    selected = torch.randint(length, (size,), generator=generator)
    marker = F.one_hot(selected, length).float()
    inputs = torch.stack([marker, values], dim=-1)
    labels = (values[torch.arange(size), selected] > 0).long()
    return inputs, labels, selected


class AttentionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
        self.value = nn.Linear(2, 16)
        self.classifier = nn.Linear(16, 2)

    def forward(self, inputs, uniform=False):
        """입력 위치의 중요도를 학습해 분류 logits와 가중치를 반환합니다.

        inputs: (B, T, 2), logits: (B, 2), weights: (B, T).
        """
        scores = self.scorer(inputs).squeeze(-1)
        # 위치별 점수를 확률로 바꾸어 어느 입력을 참고할지 정합니다.
        weights = scores.softmax(dim=-1)
        if uniform:
            weights = torch.ones_like(weights) / inputs.size(1)
        # 가중치를 value 벡터에 곱해 하나의 문맥 벡터로 모읍니다.
        context = torch.bmm(weights.unsqueeze(1), self.value(inputs)).squeeze(1)
        return self.classifier(context), weights


@torch.no_grad()
def evaluate(model, batch, uniform=False):
    """기울기를 기록하지 않고 손실과 정확도를 계산합니다.

    실습별로 선택 위치, 검색 위치 또는 답 위치도 함께 측정합니다.
    """
    model.eval()
    inputs, labels, selected = batch
    logits, weights = model(inputs, uniform=uniform)
    return {
        'loss': F.cross_entropy(logits, labels).item(),
        'accuracy': (logits.argmax(-1) == labels).float().mean().item(),
        'selected_mass': weights[torch.arange(len(labels)), selected].mean().item(),
    }


def main():
    """실행 옵션을 읽고 학습, 검증과 결과 저장을 순서대로 수행합니다."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=700)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--out', default='runs/week01')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    train_rng = torch.Generator().manual_seed(args.seed + 1)
    val_rng = torch.Generator().manual_seed(args.seed + 1000)
    validation = make_batch(2048, val_rng)
    model = AttentionClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    history = [{'step': 0, **evaluate(model, validation)}]
    before = model.scorer[0].weight.detach().clone()
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        inputs, labels, _ = make_batch(128, train_rng)
        logits, _ = model(inputs)
        # 정답과 logits를 비교합니다. 소프트맥스는 손실 함수 내부에서 처리합니다.
        loss = F.cross_entropy(logits, labels)
        # 이전 배치의 기울기를 비우고 현재 손실을 역전파합니다.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1:
            print('scorer gradient norm:', model.scorer[0].weight.grad.norm().item())
        # 기울기가 지나치게 커질 때 업데이트의 크기를 제한합니다.
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 계산된 기울기로 학습 가능한 파라미터를 갱신합니다.
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            metrics = {'step': step, 'train_loss': loss.item(), **evaluate(model, validation)}
            history.append(metrics)
            print(json.dumps(metrics))
    assert not torch.equal(before, model.scorer[0].weight)
    model.eval()
    with torch.no_grad():
        _, weights = model(validation[0][:1])
        assert torch.allclose(weights.sum(-1), torch.ones(1), atol=1e-6)
    report = {
        'torch': torch.__version__, 'seed': args.seed, 'steps': args.steps,
        'seconds': time.perf_counter() - started, 'history': history,
        'uniform_intervention': evaluate(model, validation, uniform=True),
        'sample_selected': validation[2][0].item(),
        'sample_weights': weights[0].tolist(),
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    torch.save(model.state_dict(), output / 'model.pt')
    print(json.dumps({k: v for k, v in report.items() if k != 'history'}, indent=2))


if __name__ == '__main__':
    main()
```

<a id="section-6"></a>
## 학습 결과

기본 설정은 시드 41, 700단계 학습입니다.

| 검증 지표 | 학습 전 | 마지막 단계 |
|---|---:|---:|
| 교차 엔트로피 | 0.68371 | 0.02466 |
| 정확도 | 56.54% | 99.85% |
| 표식 위치의 평균 가중치 | 0.15354 | 0.98967 |

![검증 손실과 정확도 그래프](https://teimakes.github.io/attention-course/week01-learning.svg)

학습한 모델의 가중치를 균등하게 바꿨을 때 정확도는 60.94%였습니다.

| 시드 | 마지막 단계의 검증 정확도 |
|---|---:|
| 41 | 99.85% |
| 51 | 99.66% |
| 61 | 99.76% |

[측정 기록 보기](https://teimakes.github.io/attention-course/week01-metrics.json)


`accuracy`는 분류 정확도이고 `selected_mass`는 표식 위치에 부여한 평균 가중치입니다. 두 지표를 함께 보면 최종 분류 성능과 입력 선택의 변화를 각각 확인할 수 있습니다.

학습한 모델의 어텐션 가중치를 모두 같게 바꾸는 실험도 포함했습니다. 이때 분류기가 유지된 상태에서 가중치만 바뀝니다. 균등 가중치를 사용하는 모델을 처음부터 새로 학습한 결과와는 다른 실험입니다.

<a id="section-7"></a>
## 직접 확인해 보기

1. 입력 길이를 8에서 16으로 바꾸고 같은 설정으로 학습합니다. 정확도와 표식 위치의 평균 가중치를 비교합니다.
2. 균등 가중치를 사용하는 모델을 처음부터 학습합니다. 학습 후 가중치만 바꾼 실험과 결과가 다른지 확인합니다.
3. `weights`에 `detach()`를 적용한 뒤 점수 함수의 기울기가 어떻게 바뀌는지 확인합니다. 기울기가 없는 상태에서 파라미터가 갱신되는지도 검사합니다.

다음 글에서는 질문에 따라 서로 다른 입력 위치를 선택하도록 모델을 확장합니다.

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
