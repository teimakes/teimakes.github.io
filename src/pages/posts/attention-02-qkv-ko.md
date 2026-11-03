---
layout: '../../layouts/Post.astro'
title: '어텐션 강의 2주차: Q·K·V를 학습하는 정보 검색 모델'
description: '학습 가능한 PyTorch 코드와 실습 결과를 통해 어텐션을 살펴봅니다.'
date: '2026-11-03'
tags: ['attention', 'transformer', 'pytorch', 'deep-learning']
---

> **어텐션과 트랜스포머 · 2/5**  
> 질문에 맞는 정보를 찾는 모델을 학습하며 Q·K·V의 역할을 알아봅니다.

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

- [이번 주의 질문: 같은 자료에서도 질문에 따라 읽을 곳을 바꿀 수 있을까?](#section-1)
- [1. 데이터와 정답을 먼저 명세하기](#section-2)
- [2. Q·K·V를 분리하면 무엇이 달라질까?](#section-3)
- [3. 행렬 연산을 표본 하나에서 배치로 확장하기](#section-4)
- [4. 왜 sqrt(dk)로 나누는가?](#section-5)
- [5. 검색은 어떻게 학습되는가?](#section-6)
- [6. 전체 PyTorch 코드](#section-7)
- [7. 측정 결과와 검색 실패 분석](#section-9)
- [8. 위치 순서를 바꿔도 결과가 같은 이유](#section-10)
- [9. 실습 확장과 과제](#section-11)

<a id="section-1"></a>
## 이번 주의 질문: 같은 자료에서도 질문에 따라 읽을 곳을 바꿀 수 있을까?

지난주 모델은 표식이 있는 위치를 골랐습니다. 이번에는 별도의 질문이 주어집니다. 책상 위에 이름과 값이 적힌 카드 다섯 장이 있고, ‘이 이름에 해당하는 값은 무엇인가?’라고 묻습니다. 카드 순서는 매번 바뀌고 이름에 연결된 값도 바뀝니다. 이름 하나의 정답을 외우는 방식으로는 해결할 수 없습니다.

이 실습은 Q·K·V를 **학습시켜야 하는 이유**를 드러냅니다. Query는 질문을 비교에 적합한 벡터로, Key는 카드의 이름을 비교 가능한 벡터로, Value는 찾은 카드에서 가져올 값을 표현합니다. 코드에서 모든 투영은 학습 가능한 PyTorch 계층입니다.

### 수업 구성과 목표

강의 90분은 검색 문제와 Q·K·V 30분, 행렬 계산과 스케일링 30분, 학습과 불변성 30분으로 나눕니다. 실습 90분은 실행·중간 텐서 확인·스케일 제거 실험으로 구성합니다. 과제는 새로운 카드 수와 시드에 대한 결과 분석 2~3시간입니다.

학습 목표는 `(B,Q,K)` 점수 텐서의 각 축을 설명하고, 교차 어텐션과 셀프 어텐션을 구분하며, 정답 분류 정확도와 검색 위치 정확도를 별도로 측정하는 것입니다. 지난주 소프트맥스와 교차 엔트로피를 이해했다고 가정합니다.

<a id="section-2"></a>
## 1. 데이터와 정답을 먼저 명세하기

이름 종류는 12개, 한 표본의 카드 수는 5개, 값 종류는 6개입니다. 각 표본에서는 중복되지 않는 이름 5개를 고르고, 각 이름에 0~5의 값을 무작위로 붙입니다. 질문은 이 다섯 이름 중 하나이고 정답은 해당 카드의 값입니다.

예를 들어 카드가 `[(7,2),(1,5),(9,0),(4,3),(2,1)]`이고 질문이 4라면 정답은 3입니다. 다음 표본에서는 이름 4에 전혀 다른 값이 붙을 수 있습니다. 학습과 검증에서 이름 자체를 완전히 분리하지는 않습니다. 따라서 검증은 **이미 알고 있는 이름들의 새로운 배열과 값 배치**에 대한 일반화를 측정합니다. 한 번도 등장하지 않은 이름으로의 일반화는 별도의 문제입니다.

코드는 정답 카드 위치를 반환하지만, 손실은 값 클래스 정답에만 적용합니다. 위치는 검색 정확도를 측정하는 진단 지표입니다. 값 종류가 적으므로 다른 카드를 봐도 우연히 같은 값을 얻을 수 있습니다. 이 때문에 출력 정확도만으로 검색이 올바른지 판단하면 안 됩니다.

<a id="section-3"></a>
## 2. Q·K·V를 분리하면 무엇이 달라질까?

이름 임베딩을 Eₖ, 값 임베딩을 Eᵥ라고 합시다. 질문과 카드 이름은 같은 이름 임베딩을 공유하되 서로 다른 투영을 통과합니다.

```text
Q = Eₖ(질문) Wq
K = Eₖ(카드 이름들) Wk
V = Eᵥ(카드 값들) Wv
S = Q Kᵀ / sqrt(dk)
A = softmax(S, key 축)
C = A V
logits = C Wo + b
```

Q·K는 어디를 읽을지 결정하고 V는 그곳에서 무엇을 전달할지 정합니다. 이 실습에서는 카드 이름을 K에, 값 내용을 V에 넣어 역할을 선명하게 구분했습니다. 일반적인 셀프 어텐션에서는 같은 토큰 표현 X로부터 세 투영을 모두 만듭니다. **이름과 값의 입력을 분리하는 현재 구조는 교차 어텐션 기반의 통제된 검색 모델**이지, 완성된 언어 모델은 아닙니다.

같은 임베딩에서 Q와 K를 만들더라도 Wq와 Wk는 다를 수 있습니다. 질문할 때 필요한 표현과 검색 대상으로 제시할 표현이 완전히 같아야 할 이유가 없기 때문입니다. 두 투영을 하나로 묶으면 학습 가능한 함수의 형태도 바뀝니다.

<a id="section-4"></a>
## 3. 행렬 연산을 표본 하나에서 배치로 확장하기

표본 하나에서는 Q가 `(1,32)`, K가 `(5,32)`이므로 `Q @ K.T`는 `(1,5)`입니다. 배치 128개를 한꺼번에 계산하면 Q `(128,1,32)`, K `(128,5,32)`, 점수 `(128,1,5)`가 됩니다. `transpose(-2,-1)`은 마지막 두 축만 바꾸므로 배치끼리 섞이지 않습니다.

`softmax(-1)`은 다섯 카드에 대한 비율을 계산합니다. 가중치 `(128,1,5)`와 V `(128,5,32)`를 곱하면 문맥 `(128,1,32)`입니다. 길이 1인 query 축만 `squeeze(1)`로 제거하면 분류기에 넣을 `(128,32)`를 얻습니다. 단순히 `squeeze()`를 쓰면 배치가 1일 때 배치 축까지 사라질 수 있으므로 제거할 축을 명시했습니다.

PyTorch `nn.Linear(in,out)`의 가중치 저장 모양은 `(out,in)`입니다. 논문에서 쓰는 `XW` 표기와 실제 속성 `linear.weight`의 모양이 달라도 모순이 아닙니다. 선형 계층이 내부적으로 전치를 포함한 연산을 수행합니다.

<a id="section-5"></a>
## 4. 왜 sqrt(dk)로 나누는가?

q와 k의 각 좌표가 평균 0, 분산 1이며 서로 독립적이라고 단순화하면, 내적은 dk개의 곱의 합이므로 분산이 dk에 비례합니다. 큰 점수 차이는 소프트맥스를 한 위치에 과도하게 집중시킬 수 있습니다. 포화 영역에서는 작은 점수 변화가 가중치에 거의 영향을 주지 않아 학습이 어려워질 수 있습니다.

```text
Var(Σᵢ qᵢkᵢ) ≈ dk
Std(내적) ≈ sqrt(dk)
scaled score = 내적 / sqrt(dk)
```

이것은 초기 분산을 설명하는 근사이고, 학습된 Q·K가 항상 독립·단위분산이라는 뜻은 아닙니다. 스케일링이 모든 작은 실험에서 반드시 더 높은 정확도를 보장하는 것도 아닙니다. 아래 코드의 `--no-scale` 옵션은 가정을 실제로 비교할 수 있도록 제공합니다.

<a id="section-6"></a>
## 5. 검색은 어떻게 학습되는가?

초기에는 어떤 카드의 V를 섞어야 정답이 되는지 모릅니다. 최종 분류가 틀리면 교차 엔트로피의 기울기가 읽어 온 문맥과 V, 소프트맥스, Q·K를 거쳐 임베딩까지 전달됩니다. 올바른 카드에서 값을 가져오는 규칙이 오차를 줄인다면 해당 이름끼리의 점수가 높아지는 방향으로 조정될 수 있습니다.

정답 위치의 점수를 직접 최대화하는 보조 손실은 사용하지 않습니다. 검색 정확도가 높아지는 현상과 최종 손실의 관계를 관찰하는 것이 이번 실습의 목적입니다. 위치 정답까지 손실로 주면 더 직접적인 감독이 되어 다른 실험이 됩니다.

<a id="section-7"></a>
## 6. 전체 PyTorch 코드

전체 소스는 [week02.py](/attention-course/week02.py)로도 받을 수 있습니다. 아래 코드는 같은 파일의 전체 내용입니다.

```bash
python week02.py
```

```python
"""Week 2: train Q/K/V projections on associative retrieval."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def make_batch(size, generator, slots=5, key_count=12, classes=6):
    """이름이 중복되지 않는 카드와 질문을 만들고 질문에 해당하는 값을 반환합니다.

    카드 이름과 값은 (배치, 카드 수), 질문과 정답은 각각 (배치,)입니다.
    """
    keys = torch.rand(size, key_count, generator=generator).argsort(-1)[:, :slots]
    values = torch.randint(classes, (size, slots), generator=generator)
    chosen = torch.randint(slots, (size,), generator=generator)
    rows = torch.arange(size)
    return keys, values, keys[rows, chosen], values[rows, chosen], chosen


class Retriever(nn.Module):
    def __init__(self, scale=True):
        super().__init__()
        self.scale = scale
        self.key_embedding = nn.Embedding(12, 32)
        self.value_embedding = nn.Embedding(6, 32)
        self.wq = nn.Linear(32, 32, bias=False)
        self.wk = nn.Linear(32, 32, bias=False)
        self.wv = nn.Linear(32, 32, bias=False)
        self.readout = nn.Linear(32, 6)

    def forward(self, keys, values, query, uniform=False):
        """질문 Q와 카드 K의 유사도로 값 V를 모아 정답 클래스를 예측합니다.

        keys/values: (B, T), query: (B,), logits: (B, 6).
        """
        q = self.wq(self.key_embedding(query)).unsqueeze(1)
        k = self.wk(self.key_embedding(keys))
        v = self.wv(self.value_embedding(values))
        # 각 query와 key의 내적으로 위치별 유사도를 계산합니다.
        scores = q @ k.transpose(-2, -1)
        if self.scale:
            # 내적의 크기를 key 차원의 제곱근으로 조절합니다.
            scores = scores / math.sqrt(k.size(-1))
        # 위치별 점수를 확률로 바꾸어 어느 입력을 참고할지 정합니다.
        weights = scores.softmax(-1)
        if uniform:
            weights = torch.ones_like(weights) / keys.size(1)
        # 헤드별 문맥을 이어 붙여 원래 모델 차원으로 복원합니다.
        context = (weights @ v).squeeze(1)
        return self.readout(context), weights.squeeze(1)


@torch.no_grad()
def evaluate(model, batch, uniform=False):
    """기울기를 기록하지 않고 손실과 정확도를 계산합니다.

    실습별로 선택 위치, 검색 위치 또는 답 위치도 함께 측정합니다.
    """
    model.eval()
    keys, values, query, target, chosen = batch
    logits, weights = model(keys, values, query, uniform)
    return {
        'loss': F.cross_entropy(logits, target).item(),
        'accuracy': (logits.argmax(-1) == target).float().mean().item(),
        'retrieval_accuracy': (weights.argmax(-1) == chosen).float().mean().item(),
    }


def main():
    """실행 옵션을 읽고 학습, 검증과 결과 저장을 순서대로 수행합니다."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=700)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-scale', action='store_true')
    parser.add_argument('--out', default='runs/week02')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    train_rng = torch.Generator().manual_seed(args.seed + 1)
    validation = make_batch(2048, torch.Generator().manual_seed(args.seed + 1000))
    model = Retriever(scale=not args.no_scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    history = [{'step': 0, **evaluate(model, validation)}]
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        keys, values, query, target, _ = make_batch(128, train_rng)
        logits, _ = model(keys, values, query)
        # 정답과 logits를 비교합니다. 소프트맥스는 손실 함수 내부에서 처리합니다.
        loss = F.cross_entropy(logits, target)
        # 이전 배치의 기울기를 비우고 현재 손실을 역전파합니다.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # 기울기가 지나치게 커질 때 업데이트의 크기를 제한합니다.
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 계산된 기울기로 학습 가능한 파라미터를 갱신합니다.
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            metrics = {'step': step, 'train_loss': loss.item(), **evaluate(model, validation)}
            history.append(metrics)
            print(json.dumps(metrics))
    model.eval()
    with torch.no_grad():
        keys, values, query, _, _ = validation
        before, weights = model(keys[:4], values[:4], query[:4])
        permutation = torch.tensor([4, 2, 0, 1, 3])
        after, _ = model(keys[:4, permutation], values[:4, permutation], query[:4])
        assert torch.allclose(before, after, atol=1e-5)
        assert torch.allclose(weights.sum(-1), torch.ones(4), atol=1e-6)
    report = {'torch': torch.__version__, 'seed': args.seed, 'scale': model.scale,
              'seconds': time.perf_counter() - started, 'history': history,
              'uniform_intervention': evaluate(model, validation, True)}
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    torch.save(model.state_dict(), output / 'model.pt')
    print('permutation invariance: passed')


if __name__ == '__main__':
    main()
```

<a id="section-9"></a>
## 7. 측정 결과와 검색 실패 분석

측정 환경은 Python 3.12.14, PyTorch 2.15.0+cpu, CPU, 스레드 1개입니다. 기본 시드는 42이고 마지막 기록은 700 step입니다.

| 지표 | 학습 전 | 마지막 step |
|---|---:|---:|
| 검증 교차 엔트로피 | 1.82910 | 0.00012 |
| 검증 정확도 | 14.50% | 100.00% |
| 검색 위치 정확도 | 0.13721 | 1.00000 |

![검증 손실과 정확도의 학습 곡선](/attention-course/week02-learning.svg)

[원본 측정 기록](/attention-course/week02-metrics.json) · [실험별 전체 기록](/attention-course/results.json)

학습 후 균등 가중치로 개입했을 때 정확도는 **42.68%**였습니다. 이는 동일 구조의 균등 모델을 처음부터 학습한 결과와는 구분해야 합니다.

추가 재현 실행의 **마지막 step의 고정 검증 정확도**입니다. 데이터 시드도 함께 달라집니다.

| 시드 | 정확도 |
|---|---:|
| 42 | 100.00% |
| 52 | 100.00% |
| 62 | 100.00% |



`accuracy`는 최종 값 분류 정확도이고 `retrieval_accuracy`는 가장 큰 가중치를 받은 카드가 정답 위치인지 측정합니다. 우연한 값 추측의 기준은 대략 1/6, 카드 위치의 기준은 1/5입니다. 고정 검증 표본에서는 표본 변동이 있습니다. 균등 가중치일 때 `argmax`는 첫 동률 위치를 선택하므로 검색 지표를 의미 있는 학습으로 해석해서는 안 됩니다.

공개한 값은 한 실행의 측정값입니다. 다른 시드와 버전에서는 수렴 속도와 마지막 숫자가 달라질 수 있습니다. 특히 ‘100%’는 이 고정 검증 집합에서 실패가 없었다는 뜻이며, 모든 가능한 입력에서의 수학적 정확성을 보장하지 않습니다.

<a id="section-10"></a>
## 8. 위치 순서를 바꿔도 결과가 같은 이유

카드들을 같은 순열로 섞으면 K와 V의 행이 함께 바뀝니다. 점수의 열과 가중치의 열도 같은 방식으로 바뀌므로 최종 가중합은 그대로입니다. 이 구조에는 위치 임베딩이 없고 카드의 순서 자체가 정답과 무관하기 때문입니다.

코드 마지막의 순열 검사는 이런 성질을 실제 계산에서 확인합니다. K만 섞고 V를 그대로 두면 이름과 값의 대응이 깨져 다른 입력이 됩니다. 두 종류의 순서 변경을 구분해야 합니다. 다음 주처럼 위치가 의미를 가지는 문제에서는 이 불변성이 오히려 한계가 됩니다.

<a id="section-11"></a>
## 9. 실습 확장과 과제

1. 기본 실행과 `--no-scale --out runs/week02-no-scale`를 같은 시드 세 개로 비교합니다. 최종 정확도뿐 아니라 첫 100단계 손실과 가중치 엔트로피도 측정하세요.
2. 카드 수를 5에서 10으로 늘립니다. 데이터 함수와 순열 검사 길이를 함께 수정해야 합니다. 검색 후보 증가가 학습에 미치는 영향을 설명하세요.
3. 질문을 검증 카드에 없는 이름으로 바꿉니다. 현재 모델에는 ‘없음’ 클래스를 학습하지 않았습니다. 확률이 높게 출력되더라도 답이 존재함을 보장하지 않는 이유를 설명하세요.
4. Q와 K 투영을 공유하도록 바꾼 모델을 처음부터 학습하고 파라미터 수와 성능을 비교하세요.

과제는 이론 25점, Q·K·V 모양과 배치 구현 30점, 세 시드 비교 30점, 일반화 범위와 실패 해석 15점입니다. 보고서에는 최소 다섯 실패 사례의 카드·질문·정답·예측·가중치를 함께 제시합니다. 실패가 없다면 카드 수나 학습 단계를 바꾸어 경계 조건을 찾아보세요.

## 읽을 자료

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 특히 3.2~3.5절. 구조와 수식의 원 출처입니다.
- [PyTorch SDPA 공식 문서](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html): 마스크와 스케일링의 구현 의미를 확인합니다.
- [PyTorch CrossEntropyLoss 공식 문서](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html): logits, 정수 타깃과 무시 인덱스를 확인합니다.

