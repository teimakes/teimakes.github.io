---
layout: '../../layouts/Post.astro'
title: '어텐션과 트랜스포머 2: Q·K·V와 어텐션 학습'
description: 'Query·Key·Value의 역할과 텐서 차원을 따라가며 스케일드 닷 프로덕트 어텐션과 학습 가능한 투영을 PyTorch로 구현합니다.'
date: '2026-11-03'
tags: ['attention', 'transformer', 'pytorch', 'deep-learning']
---

> **어텐션과 트랜스포머 · 2/5**

Query와 Key는 참조할 위치의 점수를 계산하고, Value는 그 가중치로 더할 정보를 나타냅니다. 세 표현을 구분하면 어텐션 수식을 코드로 구현할 때 각 행렬곱이 무엇을 계산하는지 분명해집니다.

이 시리즈는 입력 선택을 학습하는 작은 모델에서 시작해 다섯 번째 글의 디코더 언어 모델까지 구현합니다. 이번 2편에서는 질문에 해당하는 값을 찾는 문제로 Q·K·V의 투영 행렬과 임베딩을 학습합니다. 이후에는 같은 입력에서 Q·K·V를 만드는 셀프 어텐션, 미래를 가리는 마스크, 여러 헤드의 결합을 다룹니다.

코드의 목적은 값의 정답만으로 모델이 질문과 맞는 항목을 참조하도록 학습되는지 확인하는 것입니다. 질문할 때마다 이름에 연결된 값과 항목 순서가 달라지므로, 특정 이름의 값을 외우는 것으로는 해결할 수 없습니다. 최종 값 예측과 정답 위치 검색을 따로 평가해 두 단계가 어떻게 학습되는지 확인합니다.

## 목차

- [시리즈에서 다루는 내용](#section-1)
- [실습 문제](#section-2)
- [Q·K·V 계산](#section-3)
- [텐서 차원](#section-4)
- [내적을 √dₖ로 나누는 이유](#section-5)
- [어떤 파라미터가 학습되는가?](#section-6)
- [계산 순서를 바꾸면 생기는 차이](#section-7)
- [높은 정확도를 어떻게 해석할까?](#section-8)
- [전체 코드](#section-9)
- [실습 준비](#section-10)
- [학습 결과](#section-11)
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

입력에는 이름 ID와 값이 짝을 이룬 항목 다섯 개가 있습니다. 질문으로 이름 ID 하나가 주어지면 그 이름에 연결된 값을 출력합니다.

```text
항목: [(7, 2), (1, 5), (9, 0), (4, 3), (2, 1)]
질문: 4
정답: 3
```

각 표본의 이름은 중복되지 않지만, 이름에 연결되는 값은 표본마다 바뀝니다. 따라서 특정 이름의 값을 고정적으로 학습해서는 문제를 해결할 수 없습니다. 학습과 검증에서 이름 ID의 종류는 공유하고, 항목의 배열과 값은 독립적으로 생성합니다. 검증 결과는 새로운 이름을 처리하는 성능이 아니라 **기존 이름으로 만들어진 새로운 조합을 처리하는 성능**입니다.

모델은 질문을 Query, 항목의 이름을 Key, 항목의 값을 Value로 변환합니다. Query와 Key는 서로 다른 입력에서 만들어지므로 이 예제는 교차 어텐션입니다. 셀프 어텐션에서는 같은 입력으로부터 Q·K·V를 만듭니다.

<a id="section-3"></a>
## Q·K·V 계산

```text
Q = 질문 임베딩 × Wq
K = 항목 이름 임베딩 × Wk
V = 항목 값 임베딩 × Wv
점수 = QKᵀ / √dₖ
가중치 = softmax(점수)
문맥 = 가중치 × V
```

Wq, Wk, Wv는 학습 가능한 행렬입니다. Query와 Key의 마지막 차원은 내적을 할 수 있도록 같아야 합니다. Value의 차원은 Query·Key의 차원과 다르게 정할 수도 있습니다. 이 실습에서는 세 차원을 모두 32로 사용합니다.

항목별 점수는 질문과 각 이름의 내적으로 계산합니다. 소프트맥스는 다섯 항목에 대한 점수를 가중치로 바꿉니다. 각 가중치를 대응하는 Value에 곱해 더하면 질문에 맞는 문맥 벡터가 만들어집니다. 마지막 분류기는 이 벡터로 여섯 개 값 중 하나를 예측합니다.

<a id="section-4"></a>
## 텐서 차원

배치 크기를 B, 항목 수를 T라고 하면 코드에서 사용하는 모양은 다음과 같습니다.

| 텐서 | 모양 | 의미 |
|---|---|---|
| Q | `(B, 1, 32)` | 표본마다 질문 하나 |
| K | `(B, T, 32)` | 각 항목의 이름 표현 |
| V | `(B, T, 32)` | 각 항목의 값 표현 |
| 점수와 가중치 | `(B, 1, T)` | 질문과 각 항목의 관계 |
| 문맥 | `(B, 1, 32)` | Value의 가중합 |

`K.transpose(-2, -1)`은 Key의 마지막 두 축을 바꿉니다. 따라서 `(B, 1, 32) @ (B, 32, T)`의 결과가 `(B, 1, T)`가 됩니다. 소프트맥스는 마지막 T축에 적용합니다. 그다음 `(B, 1, T) @ (B, T, 32)`를 계산해 문맥 벡터를 얻습니다.

### 두 항목으로 계산해 보기

Query가 `[1, 0]`이고 두 Key가 각각 `[1, 0]`, `[0, 1]`이라면 내적은 `[1, 0]`입니다. dₖ가 2이므로 `√2`로 나눈 점수는 약 `[0.707, 0]`이고 소프트맥스 가중치는 약 `[0.670, 0.330]`입니다.

두 Value가 `[2, 0]`과 `[0, 4]`라면 문맥 벡터는 대략 `[1.340, 1.320]`입니다. Query와 Key로 가중치를 계산하고 실제로 더하는 벡터는 Value입니다. 실습 코드에서는 항목 다섯 개와 32차원 벡터를 사용하지만 계산 순서는 같습니다.

<a id="section-5"></a>
## 내적을 √dₖ로 나누는 이유

Query와 Key의 각 성분이 평균 0, 분산 1인 독립 확률변수라고 가정하면 내적의 분산은 차원 dₖ에 비례합니다. 차원이 커지면 점수의 크기도 커지는 경향이 있습니다. 점수 차이가 매우 크면 소프트맥스가 한 위치에 치우치고 기울기가 작아질 수 있습니다.

점수를 `√dₖ`로 나누는 것은 이런 점수 크기 증가를 완화하기 위한 방법입니다. 이 설명은 초기 분포에 대한 단순한 가정에 기초합니다. 학습 후의 성분들이 항상 서로 독립이라는 뜻은 아닙니다. 코드에는 스케일링을 끄는 `--no-scale` 옵션이 있으므로 같은 데이터와 시드에서 차이를 비교할 수 있습니다.

<a id="section-6"></a>
## 어떤 파라미터가 학습되는가?

이 코드에서 이름과 값은 정수 ID입니다. 숫자 ID 자체의 크기에 의미가 있는 것은 아닙니다. ID 7이 ID 1보다 더 중요하다는 관계를 가정하지 않습니다. `nn.Embedding`은 각 ID에 해당하는 행을 선택해 벡터를 반환하며, 학습은 그 벡터의 원소도 갱신합니다.

`key_embedding`은 질문과 항목 이름에 공유됩니다. 그러나 질문 벡터에는 Wq를, 항목 이름 벡터에는 Wk를 적용하므로 최종 Query와 Key가 같을 필요는 없습니다. 비교에 사용할 두 표현을 학습하면서 어떤 이름끼리 점수를 높일지도 조정합니다.

Wv와 `readout`도 학습됩니다. 올바른 항목에 높은 가중치를 주는 것만으로 최종 값 ID가 자동으로 출력되지는 않습니다. `value_embedding`과 Wv가 만든 문맥 표현을 `readout`이 여섯 클래스의 점수로 바꾸는 과정까지 함께 학습합니다.

학습 신호는 최종 값의 정답입니다. 위치 정답 `chosen`은 `loss`에 전달하지 않습니다. 따라서 정답 위치의 가중치가 커지는 현상은 값을 맞히는 학습과 함께 관찰되는 결과이지, 가중치에 위치 정답을 직접 지정한 결과가 아닙니다.

<a id="section-7"></a>
## 계산 순서를 바꾸면 생기는 차이

`weights @ V`는 각 항목의 Value를 같은 위치의 가중치로 더합니다. Key와 Value의 위치 대응을 유지해야 하는 이유입니다. Key만 다른 순서로 바꾸면 이름과 값의 관계가 달라지고 모델은 다른 문제를 계산하게 됩니다.

Key와 Value를 함께 같은 순서로 바꿀 때는 결과가 유지됩니다. 순열 행렬을 P라고 하면 `K'=PK`, `V'=PV`이며 다음처럼 정리할 수 있습니다.

```text
QK'ᵀ = QKᵀPᵀ
softmax(QK'ᵀ) = A Pᵀ
(A Pᵀ)(P V) = A V
```

여기서 A는 원래 어텐션 가중치입니다. 순열은 가중치의 항목 순서를 바꿀 뿐 최종 가중합은 바꾸지 않습니다. 코드 마지막의 검사는 이 관계를 작은 배치에서 확인합니다. 위치 임베딩이 추가되거나 항목 순서가 정답의 일부인 문제에는 이 설명을 그대로 적용할 수 없습니다.

<a id="section-8"></a>
## 높은 정확도를 어떻게 해석할까?

값의 종류는 여섯 개, 항목 수는 다섯 개입니다. 임의로 값을 고르면 기대 정확도는 약 16.7%이고, 임의로 항목 위치를 고르면 20%입니다. 서로 다른 지표에 같은 기준을 적용하면 안 됩니다.

값이 같은 항목이 둘 이상 있으면 잘못된 위치를 참조하고도 최종 값은 맞힐 수 있습니다. 반대로 검색 위치가 맞더라도 Value 표현과 출력 분류기가 아직 학습되지 않았다면 값 예측은 틀릴 수 있습니다. 두 정확도를 따로 기록하면 어느 단계에서 오류가 생겼는지 분석하기 좋습니다.

검증 데이터에는 학습에서 사용한 이름들이 다시 등장합니다. 새로운 이름 ID를 추가하려면 임베딩도 추가해야 하며, 학습하지 않은 행의 벡터만으로 검색을 잘할 것이라고 기대할 근거는 없습니다. 이 실습의 검증 범위는 같은 이름 집합에서 값과 배열을 새로 만든 경우입니다.

<a id="section-9"></a>
## 전체 코드

`Retriever`의 `forward()`에서 질문과 항목을 각각 임베딩하고 Q·K·V를 계산합니다. 학습 시 손실에는 값의 정답만 사용합니다. 정답 항목의 위치는 검색 정확도를 측정할 때 따로 사용합니다.

`key_embedding`은 질문 ID와 항목 이름 ID에 함께 사용됩니다. 두 임베딩에는 각각 Wq와 Wk가 적용됩니다. `value_embedding`은 값 ID를 표현하며 Wv를 통과합니다. 이후 문맥을 `readout`에 넣고 여섯 클래스의 logits를 얻습니다.

`keys`는 표본마다 12개 이름을 무작위 순서로 정렬한 뒤 앞의 다섯 개를 선택합니다. 따라서 같은 표본에서 이름이 중복되지 않습니다. 값은 0~5 중에서 독립적으로 선택하므로 서로 다른 항목에 같은 값이 들어갈 수 있습니다. 이 때문에 최종 출력 정확도와 정답 위치 검색 정확도를 따로 기록합니다.

<a id="section-10"></a>
## 실습 준비

Python 3.12에서 실행할 수 있습니다. PyTorch를 설치한 뒤 [week02.py](https://teimakes.github.io/attention-course/week02.py)를 내려받아 실행하세요. 아래 코드와 내려받는 파일은 같습니다.

```bash
python -m pip install torch==2.15.0
python week02.py
```

결과는 기본적으로 `runs/week02`에 저장됩니다. `--seed`, `--steps`, `--out` 옵션으로 시드, 학습 단계, 저장 경로를 지정할 수 있습니다.

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
    """이름에 연결된 값을 질문하는 검색 표본을 만듭니다.

    keys: (B, T). key_count개 이름에 무작위 점수를 부여해 정렬하고
        앞의 slots개를 선택합니다. 표본 안에서는 이름이 중복되지 않습니다.
    values: (B, T). 각 항목의 값은 classes개 클래스에서 독립적으로 뽑습니다.
    chosen: (B,). 질문할 항목의 위치를 고릅니다.
    rows: (B,). 각 표본의 행 인덱스입니다.
    keys[rows, chosen]: (B,). 질문할 이름 ID입니다.
    values[rows, chosen]: (B,). 그 이름에 연결된 정답 값입니다.

    반환한 chosen은 위치 검색 정확도에만 사용합니다. 학습은 값 정답으로
    수행하므로 모델이 어떤 항목을 참조해야 하는지 직접 감독하지 않습니다.
    """
    keys = torch.rand(size, key_count, generator=generator).argsort(-1)[:, :slots]
    values = torch.randint(classes, (size, slots), generator=generator)
    chosen = torch.randint(slots, (size,), generator=generator)
    rows = torch.arange(size)
    return keys, values, keys[rows, chosen], values[rows, chosen], chosen


class Retriever(nn.Module):
    def __init__(self, scale=True):
        """이름·값 임베딩과 Q·K·V 투영, 분류 계층을 만듭니다.

        key_embedding은 12개 이름을 각각 32차원으로 표현합니다.
        질문과 항목 이름은 이 임베딩을 공유합니다.
        value_embedding은 값 클래스 6개를 각각 32차원으로 표현합니다.
        wq와 wk는 이름 표현을 비교할 Query와 Key로 변환합니다.
        wv는 값 표현을 가중합할 Value로 변환합니다.
        readout은 문맥 벡터에서 여섯 클래스의 logits를 계산합니다.

        scale=False이면 점수를 sqrt(32)로 나누는 연산을 생략합니다.
        """
        super().__init__()
        self.scale = scale
        self.key_embedding = nn.Embedding(12, 32)
        self.value_embedding = nn.Embedding(6, 32)
        self.wq = nn.Linear(32, 32, bias=False)
        self.wk = nn.Linear(32, 32, bias=False)
        self.wv = nn.Linear(32, 32, bias=False)
        self.readout = nn.Linear(32, 6)

    def forward(self, keys, values, query, uniform=False):
        """질문과 항목의 Q·K 유사도로 V를 가중합해 값을 예측합니다.

        keys와 values는 (B, T), query는 (B,) 정수 ID입니다.

        q: 이름 임베딩에 wq를 적용하고 길이 1인 질문 축을 추가합니다.
           모양은 (B, 1, 32)입니다.
        k: 항목 이름에 wk를 적용한 (B, T, 32) 텐서입니다.
        v: 항목 값에 wv를 적용한 (B, T, 32) 텐서입니다.
        scores: q @ k.transpose(-2, -1)로 (B, 1, T)를 계산합니다.
           scale이 켜져 있으면 Key 차원의 제곱근으로 나눕니다.
        weights: 마지막 항목 축을 softmax로 정규화합니다.
           uniform=True이면 학습한 비율 대신 1/T를 사용합니다.
        context: weights @ v의 결과 (B, 1, 32)에서 질문 축만 제거합니다.
        readout: (B, 32)를 (B, 6) logits로 변환합니다.

        logits와 질문 축을 제거한 (B, T) weights를 반환합니다.
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
    """값 예측과 정답 위치 검색을 구분해 평가합니다.

    loss는 값 클래스에 대한 교차 엔트로피입니다.
    accuracy는 여섯 값 중 예측한 값이 target과 일치한 비율입니다.
    retrieval_accuracy는 최대 가중치 위치가 chosen과 같은 비율입니다.
    서로 다른 항목의 값이 같을 수 있어 두 정확도는 다를 수 있습니다.

    평가 모드와 no_grad를 사용하며, uniform=True인 개입도 지원합니다.
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
    """검색 모델을 학습하고 값 예측과 항목 선택을 평가합니다.

    --no-scale은 Q·K 점수 스케일링을 끕니다. 나머지 실행 설정은
    --steps, --seed, --out으로 지정합니다.
    검증 2,048개는 독립 생성기로 한 번 만들고 학습 내내 유지합니다.
    학습 배치 128개에서는 값 정답에 대한 교차 엔트로피만 사용합니다.
    AdamW 갱신 전 기울기를 비우고 역전파·기울기 클리핑을 수행합니다.
    100단계마다 값 정확도와 위치 검색 정확도를 함께 기록합니다.

    학습 후 이름과 값의 항목 순서를 같은 순열로 바꾸어도 logits가
    유지되는지 확인합니다. 가중치 합도 검사하고, 균등 가중치 개입과
    학습 이력을 metrics.json에, 모델 상태를 model.pt에 저장합니다.
    """
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

<a id="section-11"></a>
## 학습 결과

기본 설정은 시드 42, 700단계 학습입니다.

| 검증 지표 | 학습 전 | 마지막 단계 |
|---|---:|---:|
| 교차 엔트로피 | 1.82910 | 0.00012 |
| 정확도 | 14.50% | 100.00% |
| 검색 위치 정확도 | 13.72% | 100.00% |

![검증 손실과 정확도 그래프](https://teimakes.github.io/attention-course/week02-learning.svg)

학습한 모델의 가중치를 균등하게 바꿨을 때 정확도는 42.68%였습니다.

| 시드 | 마지막 단계의 검증 정확도 |
|---|---:|
| 42 | 100.00% |
| 52 | 100.00% |
| 62 | 100.00% |

[측정 기록 보기](https://teimakes.github.io/attention-course/week02-metrics.json)


`accuracy`는 출력한 값이 정답과 일치한 비율이고, `retrieval_accuracy`는 가중치가 가장 높은 항목이 정답 위치인 비율입니다. 두 지표는 항상 같지 않습니다. 다른 항목을 선택하더라도 우연히 값이 같으면 최종 값은 맞힐 수 있기 때문입니다.

이 모델에는 항목의 순서를 나타내는 위치 정보가 없습니다. 이름과 값을 함께 같은 순서로 섞으면 가중치의 순서도 함께 바뀌고 최종 가중합은 유지됩니다. 코드 마지막에서 이 성질을 검사합니다. 이름만 섞고 값을 그대로 두는 경우는 이름과 값의 관계 자체가 달라지므로 같은 입력이 아닙니다.

<a id="section-12"></a>
## 직접 확인해 보기

1. 기본 설정과 `--no-scale` 설정을 각각 세 시드로 실행합니다. 마지막 정확도뿐 아니라 학습 초반의 손실도 비교합니다.
2. 항목 수를 늘린 뒤 학습합니다. 데이터 생성과 순서 변경 검사가 새 항목 수에 맞는지도 확인합니다.
3. 질문에 해당하는 이름이 입력에 없는 경우를 실험합니다. 현재 모델은 ‘해당 항목 없음’을 정답으로 학습하지 않았으므로, 출력 확률이 높다는 사실만으로 정답 항목이 존재한다고 판단할 수 없습니다.

다음 글에서는 Q·K·V를 같은 시퀀스에서 만들고, 현재 위치에서 미래 위치를 참조하지 못하도록 마스크를 적용합니다.

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
