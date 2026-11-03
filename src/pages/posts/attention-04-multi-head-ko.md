---
layout: '../../layouts/Post.astro'
title: '어텐션 강의 4주차: 멀티헤드 학습과 통제된 비교 실험'
description: '학습 가능한 PyTorch 코드와 실습 결과를 통해 어텐션을 살펴봅니다.'
date: '2026-11-03'
tags: ['attention', 'transformer', 'pytorch', 'deep-learning']
---

> **어텐션과 트랜스포머 · 4/5**  
> 멀티헤드 어텐션을 구현하고 헤드 수에 따른 학습 결과를 비교합니다.

## 강의 시리즈

1. [어텐션 강의 1주차: 선택 가중치를 학습하는 원리](/posts/attention-01-big-picture-ko/)
2. [어텐션 강의 2주차: Q·K·V를 학습하는 정보 검색 모델](/posts/attention-02-qkv-ko/)
3. [어텐션 강의 3주차: 인과적 셀프 어텐션과 정보 누출](/posts/attention-03-causal-mask-ko/)
4. [어텐션 강의 4주차: 멀티헤드 학습과 통제된 비교 실험](/posts/attention-04-multi-head-ko/)
5. [어텐션 강의 5주차: 작은 언어 모델의 학습·평가·생성](/posts/attention-05-transformer-block-ko/)

## 실행 준비

Python 3.12 환경에서 CPU로 검증했습니다. 새 가상환경을 만든 뒤 PyTorch를 설치하세요. 각 주차 파일은 다른 주차 파일을 import하지 않고 독립 실행됩니다. 외부 API 키나 데이터 다운로드는 필요하지 않습니다.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch==2.15.0
```

Linux/Windows에서 CPU 전용 휠을 원하면 마지막 명령에 `--index-url https://download.pytorch.org/whl/cpu`를 붙일 수 있습니다. 글의 측정은 `2.15.0+cpu`에서 수행했습니다. [실습 안내](/attention-course/README.md)와 [실행 환경 파일](/attention-course/requirements.txt)도 함께 제공합니다.

## 이 글의 목차

- [이번 주의 질문: 두 가지 관계를 동시에 읽으려면?](#section-1)
- [1. 이번 데이터는 왜 더 어려운가?](#section-2)
- [2. 한 헤드가 평균을 만들 때 생기는 문제](#section-3)
- [3. 투영 행렬 하나와 헤드별 행렬의 관계](#section-4)
- [4. 축 변환을 종이에 적어 보기](#section-5)
- [5. 두 답의 손실과 평가](#section-6)
- [6. 전체 PyTorch 코드](#section-7)
- [7. 학습과 헤드 수 비교 결과](#section-9)
- [8. 절제 실험을 설계하기](#section-10)
- [9. 과제와 토론](#section-11)

<a id="section-1"></a>
## 이번 주의 질문: 두 가지 관계를 동시에 읽으려면?

한 번의 가중합은 여러 값 벡터를 하나로 섞습니다. 질문이 하나의 카드 값만 요구한다면 충분할 수 있습니다. 하지만 ‘민수 카드의 색’과 ‘지수 카드의 모양’을 동시에 요구한다면, 서로 다른 두 위치의 서로 다른 정보를 보존해야 합니다. 멀티헤드는 입력을 여러 투영 공간에서 읽고 그 결과를 결합하는 방법입니다.

이번 주에는 단순히 `reshape`를 호출하고 끝내지 않습니다. 두 답을 함께 맞혀야 하는 검색 모델을 학습하고, 헤드 수를 바꾼 비교 실험을 수행합니다. 헤드 수가 늘면 무조건 좋아진다는 주장을 목표로 하지 않습니다. 동일한 데이터와 총 차원에서 어떤 결과를 얻었는지 직접 측정합니다.

### 수업 계획과 목표

강의 90분: 단일 가중합의 병목 25분, 멀티헤드 수식과 축 변환 40분, 비교 실험 설계 25분. 실습 90분: 두 검색 과제 학습, 헤드별 가중치 관찰, 1헤드와 2헤드 비교. 과제 3시간: 세 시드 실험과 파라미터 수를 통제한 보고서.

목표는 `(B,H,Q,K)` 가중치 텐서의 의미를 설명하고, `reshape`와 `transpose`를 혼동하지 않으며, 항목별 정확도와 공동 정확도의 차이를 해석하는 것입니다. 2주차 검색 문제와 배치 행렬곱을 선행 지식으로 사용합니다.

<a id="section-2"></a>
## 1. 이번 데이터는 왜 더 어려운가?

각 카드에는 이름, 색, 모양이 있습니다. 이름은 12종 중 중복 없이 다섯 개를 고르고, 색과 모양은 각각 0~3에서 무작위로 정합니다. 질문에는 이름 두 개가 들어갑니다. 첫 이름의 색과 두 번째 이름의 모양을 순서대로 출력하는 것이 정답입니다. 같은 이름을 두 번 물을 수도 있습니다.

예를 들어 `민수=(색2,모양0)`, `지수=(색1,모양3)`이라면 `[민수,지수]` 질문의 답은 `[2,3]`입니다. 두 이름의 위치도 속성도 매번 바뀝니다. 이름에 대응하는 속성을 고정적으로 외우는 문제는 아닙니다.

Query 입력은 두 이름 임베딩을 이어 붙인 것입니다. Key는 카드 이름으로, Value는 카드의 색과 모양 임베딩을 이어 붙여 만듭니다. 질문·기억이 분리된 교차 어텐션 문제이므로 인과 마스크는 필요하지 않습니다. 3주차의 인과적 셀프 어텐션과 목적이 다르다는 점을 기억하세요.

<a id="section-3"></a>
## 2. 한 헤드가 평균을 만들 때 생기는 문제

하나의 헤드에서는 모든 V 차원에 같은 위치 가중치가 적용됩니다. 카드 A의 색 차원은 A만, 카드 B의 모양 차원은 B만 읽는 식으로 위치 가중치를 차원마다 독립적으로 정하지는 못합니다. 물론 충분한 표현과 비선형 계층을 주면 단일 헤드도 다양한 해법을 찾을 수 있으므로 이것을 절대적인 불가능성으로 단정해서는 안 됩니다.

멀티헤드는 서로 다른 Q·K·V 투영을 통해 위치 가중치를 여러 벌 만들 수 있습니다. 각 헤드의 문맥을 이어 붙인 뒤 출력 투영 Wo로 섞습니다.

```text
headₕ = softmax(QₕKₕᵀ / sqrt(dh)) Vₕ
merged = Concat(head₁, …, headH)
output = merged Wo
```

각 헤드가 반드시 ‘색 전담’, ‘모양 전담’이라는 사람이 정한 의미를 갖는 것은 아닙니다. 같은 목표를 여러 헤드가 협력해 표현할 수도 있고, 헤드 번호를 서로 바꾸어도 같은 함수가 될 수 있습니다. 따라서 가중치 그림에 의미를 붙일 때는 데이터와 개입 실험으로 확인해야 합니다.

<a id="section-4"></a>
## 3. 투영 행렬 하나와 헤드별 행렬의 관계

코드는 큰 Wq·Wk·Wv로 투영한 뒤 마지막 차원을 H개 조각으로 나눕니다. 큰 행렬의 출력 열들을 헤드별로 나눈 것으로 생각하면, 헤드마다 별도 투영 행렬을 둔 수식과 연결됩니다. 효율적인 구현을 위한 표현 방식의 차이입니다.

여기서는 `D=32`, `H=2`, `dh=16`입니다. H를 1로 바꾸면 각 헤드 차원이 32가 됩니다. Wq·Wk·Wv의 전체 출력 차원은 그대로이므로 코드의 파라미터 수는 동일합니다. 헤드 수를 늘리면서 D도 함께 늘린 실험과 구분할 수 있습니다.

<a id="section-5"></a>
## 4. 축 변환을 종이에 적어 보기

입력 query는 `(B,1,64)`이고 투영 후 `(B,1,32)`입니다. 이를 `(B,1,2,16)`으로 나누고 `(B,2,1,16)`으로 전치합니다. Key와 Value는 다섯 카드이므로 `(B,2,5,16)`입니다.

`Q @ K.transpose(-2,-1)`은 `(B,2,1,5)`의 점수입니다. 첫 두 축 B와 H는 독립된 행렬곱을 묶는 배치 축처럼 동작합니다. `softmax(-1)`은 카드 다섯 개를 정규화하며 헤드끼리 경쟁시키지 않습니다.

문맥은 `(B,2,1,16)`입니다. 토큰 축을 앞으로 돌려 `(B,1,2,16)`으로 만든 뒤 `(B,32)`로 이어 붙입니다. `transpose`는 메모리의 해석 순서와 stride를 바꾸므로, `contiguous()`로 연속 배열을 만든 뒤 reshape하는 과정을 명시했습니다. `reshape`가 필요하면 복사를 수행할 수도 있지만, 이 줄은 의도를 분명히 보여 줍니다.

<a id="section-6"></a>
## 5. 두 답의 손실과 평가

모델의 최종 logits는 `(B,2,4)`입니다. 두 답에 네 클래스가 있다는 뜻입니다. 이를 `(B×2,4)`로, 라벨 `(B,2)`는 `(B×2,)`로 펴서 교차 엔트로피에 전달합니다. 결과는 모든 답 위치에 대해 평균한 손실입니다.

항목별 정확도는 두 답 중 몇 개를 맞혔는지의 평균이고, 공동 정확도는 한 표본에서 두 답을 모두 맞힌 비율입니다. 독립 균등 추측이라면 항목별 기준은 1/4, 공동 기준은 1/16입니다. 모델의 실제 오차가 독립적이라는 뜻은 아니므로 항목별 정확도를 제곱해 공동 정확도를 대신해서는 안 됩니다.

<a id="section-7"></a>
## 6. 전체 PyTorch 코드

전체 소스는 [week04.py](/attention-course/week04.py)로도 받을 수 있습니다. 아래 코드는 같은 파일의 전체 내용입니다.

```bash
python week04.py
```

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

<a id="section-9"></a>
## 7. 학습과 헤드 수 비교 결과

측정 환경은 Python 3.12.14, PyTorch 2.15.0+cpu, CPU, 스레드 1개입니다. 기본 시드는 44이고 마지막 기록은 1200 step입니다.

| 지표 | 학습 전 | 마지막 step |
|---|---:|---:|
| 검증 교차 엔트로피 | 1.41209 | 0.00002 |
| 검증 정확도 | 22.27% | 100.00% |
| 두 답 공동 정확도 | 0.04541 | 1.00000 |

![검증 손실과 정확도의 학습 곡선](/attention-course/week04-learning.svg)

[원본 측정 기록](/attention-course/week04-metrics.json) · [실험별 전체 기록](/attention-course/results.json)

학습 후 균등 가중치로 개입했을 때 정확도는 **48.85%**였습니다. 이는 동일 구조의 균등 모델을 처음부터 학습한 결과와는 구분해야 합니다.

같은 시드·학습 단계에서 1헤드 모델의 공동 정확도는 46.48%, 2헤드는 100.00%였습니다. 파라미터 수는 각각 7,080개와 7,080개입니다. 단일 시드의 비교만으로 일반적인 우열을 확정하지는 않습니다.

추가 재현 실행의 **마지막 step의 고정 검증 정확도**입니다. 데이터 시드도 함께 달라집니다.

| 시드 | 정확도 |
|---|---:|
| 44 | 100.00% |
| 54 | 100.00% |
| 64 | 100.00% |



이 측정에서 H=1과 H=2는 같은 총 표현 차원과 파라미터 수를 사용했습니다. 그러나 여러 시드를 평균하기 전의 차이를 보편적 우열로 해석해서는 안 됩니다. 데이터 자체가 두 관계를 동시에 읽도록 설계되어 있어 특정 구조에 유리할 수 있습니다. 결과는 이 통제 과제에서의 증거입니다.

가중치를 읽을 때 `sample_chosen`의 두 위치와 각 헤드의 최대 위치를 비교하세요. 첫 헤드가 항상 첫 정답을, 둘째가 항상 둘째 정답을 고른다고 가정하지 마세요. 헤드들의 역할이 바뀔 수도 있습니다. 색·모양의 출력은 결합된 문맥을 다시 섞은 결과이므로 헤드 하나와 클래스 하나를 단순히 동일시하기 어렵습니다.

<a id="section-10"></a>
## 8. 절제 실험을 설계하기

균등 가중치 개입은 이미 학습한 모델의 선택 계산을 바꿉니다. 헤드 수 비교는 서로 다른 모델을 각각 처음부터 학습합니다. 두 실험은 서로 다른 질문을 다룹니다. 실험 이름에 ‘개입’과 ‘재학습 비교’를 분명히 구분하세요.

추가로 헤드 하나를 0으로 만들 수 있습니다. 다만 그 결과는 학습 시 보지 못한 내부 상태이므로 성능 저하가 그 헤드의 의미를 완전히 설명하지는 않습니다. 어느 헤드가 어떤 입력 조합에 민감한지를 찾는 출발점으로 사용하세요.

<a id="section-11"></a>
## 9. 과제와 토론

1. `--heads 1`, `--heads 2`, `--heads 4`를 세 시드씩 실행하고 공동 정확도의 평균과 범위를 보고합니다. 출력 경로를 다르게 지정해 결과를 덮어쓰지 마세요.
2. 전체 파라미터 수를 계산해 코드의 출력과 비교합니다. 임베딩, 투영, 분류기의 파라미터가 각각 얼마나 차지하는지 적습니다.
3. 질문을 같은 이름 두 번으로 제한한 집합과 다른 이름 두 개로 제한한 집합에서 성능을 따로 계산하세요.
4. D를 고정한 헤드 수 비교와 dh를 고정해 D까지 커지는 비교가 왜 다른 실험인지 설명합니다.

평가는 텐서 축 설명 30점, 완전한 학습·평가 실행 25점, 통제된 시드·헤드 비교 30점, 해석과 한계 15점입니다. 보고서에는 한 헤드가 언제나 특정 언어 기능을 담당한다는 근거 없는 이름을 붙이지 않습니다.

다음 주에는 이 연산을 디코더 블록에 넣고, 토큰화부터 자기회귀 생성까지 전체 학습 경로를 완성합니다.

## 읽을 자료

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 특히 3.2~3.5절. 구조와 수식의 원 출처입니다.
- [PyTorch SDPA 공식 문서](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html): 마스크와 스케일링의 구현 의미를 확인합니다.
- [PyTorch CrossEntropyLoss 공식 문서](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html): logits, 정수 타깃과 무시 인덱스를 확인합니다.

