---
layout: '../../layouts/Post.astro'
title: '어텐션과 트랜스포머 3: 인과적 셀프 어텐션'
description: '미래 토큰을 가리는 인과 마스크의 원리를 배우고 지연 복사 과제로 셀프 어텐션의 정보 누출과 패딩 처리를 점검합니다.'
date: '2026-11-03'
tags: ['attention', 'transformer', 'pytorch', 'deep-learning']
---

> **어텐션과 트랜스포머 · 3/5**

다음 토큰을 예측하는 모델은 현재까지 주어진 입력만 사용해야 합니다. 학습 중에는 정답 문장 전체가 있으므로, 행렬 연산을 구현할 때 어떤 위치가 어떤 입력을 참조할 수 있는지 명확히 제한해야 합니다. 인과적 셀프 어텐션은 미래 위치의 가중치를 0으로 만드는 마스크를 사용합니다.

이 시리즈는 학습 가능한 가중합에서 시작해 Q·K·V와 멀티헤드를 구현하고, 마지막에는 디코더 언어 모델을 학습합니다. 이번 3편은 자기회귀 조건을 코드에서 지키는 방법을 다룹니다. Q·K·V를 같은 시퀀스에서 계산한다는 점에서 앞 글의 검색 모델과도 다릅니다.

실습에서는 마지막 QUERY 뒤에 입력의 첫 숫자를 출력하도록 모델을 학습합니다. 목적은 정확도를 높이는 것과 함께, 미래 정보를 참조하지 않는 구현인지 확인하는 것입니다. 미래 입력을 바꿔도 그보다 앞의 출력이 유지되는지, 배치에 다른 표본을 추가해도 같은 표본의 결과가 유지되는지를 검사합니다.

## 목차

- [시리즈에서 다루는 내용](#section-1)
- [입력과 정답의 관계](#section-2)
- [지연 복사 문제](#section-3)
- [셀프 어텐션과 위치 정보](#section-4)
- [인과적 마스크](#section-5)
- [QUERY 위치의 계산을 구체적으로 확인하기](#section-6)
- [손실 위치와 기울기](#section-7)
- [정확도와 별도로 검사할 것](#section-8)
- [입력 길이를 바꾸는 실험](#section-9)
- [전체 코드](#section-10)
- [실습 준비](#section-11)
- [학습 결과와 마스크 검사](#section-12)
- [직접 확인해 보기](#section-13)

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
## 입력과 정답의 관계

다음 토큰 학습에서는 입력과 정답을 한 위치씩 어긋나게 배치합니다.

```text
입력: A B C
정답: B C D
```

첫 번째 출력은 A를 보고 B를 예측하고, 두 번째 출력은 A와 B를 보고 C를 예측합니다. 따라서 입력 위치 i는 자기 자신과 앞쪽 위치를 볼 수 있지만 i보다 뒤의 입력은 볼 수 없습니다. 이 범위를 어텐션의 마스크로 제한합니다.

<a id="section-3"></a>
## 지연 복사 문제

실습 입력은 숫자 일곱 개와 마지막 QUERY 기호로 구성합니다. QUERY 다음에 나올 정답은 첫 번째 숫자입니다.

```text
입력: 3 6 1 3 0 7 2 QUERY
정답: 3
```

`make_batch()`는 0부터 7까지의 숫자를 무작위로 선택하고, 마지막 위치에 QUERY를 나타내는 ID 8을 넣습니다. 라벨은 마지막 위치를 제외하고 모두 `-100`입니다. PyTorch의 교차 엔트로피는 기본 설정에서 `-100`인 위치의 손실을 무시합니다.

손실에서 어떤 출력을 제외하는 것과 입력 위치를 어텐션에서 가리는 것은 다릅니다. 첫 번째 위치에는 직접 손실을 계산하지 않지만, 마지막 위치의 예측이 첫 번째 위치를 참조할 수 있으므로 해당 입력과 관련된 파라미터에도 기울기가 전달됩니다.

이번 실습은 QUERY 위치의 정답 하나만 학습합니다. 모든 위치에서 다음 토큰을 학습하는 언어 모델은 5편에서 구현합니다.

<a id="section-4"></a>
## 셀프 어텐션과 위치 정보

입력 ID는 토큰 임베딩으로, 위치 번호는 위치 임베딩으로 바꾼 뒤 더합니다. 위치 정보가 필요한 이유는 정답이 숫자의 값뿐 아니라 **첫 번째 위치에 있는지**에 따라 결정되기 때문입니다.

코드의 `qkv`는 입력을 세 배 크기로 투영한 뒤 마지막 차원을 셋으로 나눕니다. 각 조각이 Q, K, V입니다. 세 표현이 모두 같은 입력에서 만들어진다는 점에서 2편의 교차 어텐션과 다릅니다.

```text
X = token_embedding(ids) + position_embedding(positions)
Q, K, V = linear(layer_norm(X))를 세 부분으로 분리
scores = QKᵀ / √dₖ
weights = softmax(scores + mask)
output = weights × V
```

이 모델은 단일 헤드를 사용합니다. 어텐션 결과에 입력을 더하는 잔차 연결이 있고, 별도의 피드포워드 계층은 없습니다.

<a id="section-5"></a>
## 인과적 마스크

점수 행렬에서 행은 현재 예측을 계산하는 Query 위치이고 열은 참조 대상인 Key 위치입니다. 길이가 3이면 허용되는 위치는 다음과 같습니다.

```text
                Key 0  Key 1  Key 2
Query 0          허용    금지    금지
Query 1          허용    허용    금지
Query 2          허용    허용    허용
```

코드는 대각선 위를 `True`로 표시한 뒤 `masked_fill()`로 그 위치의 점수를 음의 무한대로 바꿉니다. 소프트맥스에서 음의 무한대에 해당하는 가중치는 0이 됩니다. 마스크는 소프트맥스 전에 적용해야 합니다. 소프트맥스 후에 가중치만 0으로 바꾸면 남은 가중치의 합이 1이 아닐 수 있습니다.

마스크의 불리언 값은 API마다 의미가 다를 수 있습니다. 이 코드의 `masked_fill()`에서는 `True`가 바꿀 위치를 뜻합니다. PyTorch의 `scaled_dot_product_attention()`에 전달하는 불리언 마스크는 `True`가 허용 위치를 뜻하므로 API를 바꿀 때 확인해야 합니다.

### 학습할 때와 생성할 때

학습할 때는 정답 시퀀스가 이미 있으므로 모든 입력 위치의 점수를 한 번에 계산할 수 있습니다. 단, 마스크를 통해 각 위치에 허용된 정보만 전달합니다. 이를 통해 병렬 계산과 자기회귀 조건을 함께 만족시킵니다.

생성할 때는 다음 토큰이 아직 정해져 있지 않습니다. 현재까지 생성한 토큰으로 다음 토큰을 예측하고, 선택한 토큰을 입력에 추가한 다음 다시 예측합니다. 입력을 계속 늘리며 매번 전체를 계산할 수도 있고 이전 Key와 Value를 저장해 재사용할 수도 있습니다.

### 패딩과 인과 마스크

이번 데이터는 모든 입력의 길이가 같아 패딩이 필요하지 않습니다. 서로 다른 길이의 문장을 한 배치로 만들면 실제 토큰이 아닌 패딩 위치도 구분해야 합니다. 인과 마스크는 미래를 차단하고 패딩 마스크는 실제 입력이 아닌 위치를 제외합니다. 두 마스크는 목적이 다르며 실제 배치에서는 함께 사용될 수 있습니다.

<a id="section-6"></a>
## QUERY 위치의 계산을 구체적으로 확인하기

기본 입력 길이는 8입니다. 마지막 QUERY의 Query 벡터와 여덟 Key의 내적을 계산하면 점수 여덟 개를 얻습니다. 마지막 위치에서는 모든 입력이 현재 또는 과거 위치이므로 여덟 위치가 모두 허용됩니다. 모델은 그중 정답을 결정하는 첫 위치를 참조하도록 학습합니다.

첫 위치에서는 첫 Key만 허용됩니다. 따라서 첫 행의 가중치는 `[1, 0, …, 0]`입니다. 두 번째 위치에서는 첫 번째와 두 번째 Key만 허용되고, 나머지 가중치는 0입니다. 각 행은 서로 다른 범위에서 정규화되지만 행별 합은 모두 1입니다.

모든 행이 최소한 자기 위치는 포함하므로 이 코드에서는 허용된 위치가 없는 행이 생기지 않습니다. 일반적인 마스크 구현에서 한 행을 전부 음의 무한대로 만들면 소프트맥스가 유효한 분포를 만들지 못할 수 있습니다. 마스크와 패딩을 함께 적용할 때는 이런 행이 생기는지 확인해야 합니다.

<a id="section-7"></a>
## 손실 위치와 기울기

`labels`의 마지막 위치만 정답 숫자이고 나머지는 -100입니다. 출력 텐서는 모든 위치에 대해 계산하지만 교차 엔트로피는 마지막 위치만 평균에 포함합니다. 라벨을 무시한다고 해서 해당 위치의 임베딩이나 Key·Value의 기울기가 사라지는 것은 아닙니다.

마지막 QUERY의 출력은 `weights @ V`를 통해 앞 위치의 Value에 의존합니다. 그 손실을 역전파하면 참조한 위치의 Value 변환과 임베딩도 갱신됩니다. Query·Key 점수에 대한 기울기는 가중치가 달라질 때 손실이 어떻게 달라지는지를 반영합니다.

이 관계는 이번 과제의 목적과 직접 연결됩니다. 모델은 중간 숫자를 맞히도록 학습하지 않습니다. QUERY 뒤에 첫 숫자를 출력하는 규칙을 학습합니다. 따라서 다른 위치의 logits가 부정확하더라도 이번 과제의 손실과 모순되지는 않습니다.

<a id="section-8"></a>
## 정확도와 별도로 검사할 것

미래 위치의 가중치가 0인지 검사하면 마스크가 적용됐는지 확인할 수 있습니다. 그러나 어텐션 외의 연산에서 위치를 섞었다면 그 검사만으로 충분하지 않을 수 있습니다. 그래서 입력을 실제로 바꾼 뒤 logits를 비교하는 검사도 수행합니다.

마지막 두 입력만 바꿨다면 그보다 앞의 출력은 그대로여야 합니다. 바뀐 위치의 출력까지 같아야 하는 것은 아닙니다. 검사할 범위를 잘못 잡으면 올바른 모델이 실패했다고 판단할 수 있습니다.

같은 표본을 단독으로 계산한 결과와 여러 표본의 배치에서 계산한 결과도 비교합니다. 배치 축을 위치 축과 혼동하면 한 표본이 다른 표본의 정보를 사용할 수 있습니다. 이 오류는 단일 표본만으로 코드를 실행할 때 드러나지 않을 수 있습니다.

<a id="section-9"></a>
## 입력 길이를 바꾸는 실험

`--length 12`는 길이 12의 데이터로 모델을 다시 학습합니다. 길이 8로 학습한 모델을 그대로 더 긴 입력에 평가하는 실험은 아닙니다. 학습 가능한 위치 임베딩을 사용하므로 학습하지 않은 위치가 추가됐을 때의 동작은 별도로 확인해야 합니다.

길이를 늘리면 정답 위치와 QUERY 사이의 거리가 늘고 점수 행렬 크기도 증가합니다. 정확도가 유지되는지와 계산 비용이 어떻게 변하는지를 각각 측정하면 구조의 동작과 학습 결과를 구분해 볼 수 있습니다.

<a id="section-10"></a>
## 전체 코드

<a id="section-11"></a>
## 실습 준비

Python 3.12에서 실행할 수 있습니다. PyTorch를 설치한 뒤 [week03.py](https://teimakes.github.io/attention-course/week03.py)를 내려받아 실행하세요. 아래 코드와 내려받는 파일은 같습니다.

```bash
python -m pip install torch==2.15.0
python week03.py
```

결과는 기본적으로 `runs/week03`에 저장됩니다. `--seed`, `--steps`, `--out` 옵션으로 시드, 학습 단계, 저장 경로를 지정할 수 있습니다.

```python
"""Week 3: train causal self-attention for delayed copy."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def make_batch(size, generator, length=8):
    """QUERY 다음에 첫 번째 기호를 복사하는 표본을 만듭니다.

    inputs는 (B, T)이며 0~7의 정수 기호를 무작위로 생성합니다.
    마지막 입력은 전용 QUERY 기호 ID인 8로 바꿉니다.
    labels는 같은 모양으로 만들고 처음에는 모두 -100을 넣습니다.
    QUERY 위치의 라벨만 첫 번째 입력 기호로 바꿉니다.

    cross_entropy는 -100 위치의 출력 손실을 무시합니다. 해당 위치가
    입력에서 제거되는 것은 아니며 QUERY가 그 표현을 참조할 수 있습니다.
    """
    inputs = torch.randint(8, (size, length), generator=generator)
    inputs[:, -1] = 8  # QUERY marker
    labels = torch.full_like(inputs, -100)
    labels[:, -1] = inputs[:, 0]  # answer after QUERY copies first symbol
    return inputs, labels


class CausalCopy(nn.Module):
    def __init__(self, context=12, dim=32):
        """지연 복사를 학습할 단일 헤드 셀프 어텐션을 구성합니다.

        token은 숫자 8종과 QUERY를 dim차원으로 표현합니다.
        position은 최대 context개 위치의 학습 가능한 임베딩입니다.
        norm은 각 토큰의 특성 축을 정규화합니다.
        qkv는 dim차원 입력을 3*dim으로 변환해 Q·K·V를 함께 계산합니다.
        output은 각 위치의 표현을 정답 숫자 8개의 logits로 바꿉니다.

        이 모델에는 별도의 FFN이나 여러 헤드를 추가하지 않습니다.
        """
        super().__init__()
        self.token = nn.Embedding(9, dim)
        self.position = nn.Embedding(context, dim)
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, 8)

    def forward(self, ids):
        """미래 위치를 차단하고 각 위치의 숫자 logits를 계산합니다.

        ids는 (B, T) 정수 입력입니다.
        x는 토큰 임베딩과 위치 임베딩을 더한 (B, T, D) 표현입니다.
        qkv(norm(x)).chunk(3, dim=-1)는 Q·K·V를 각각 (B, T, D)로 나눕니다.
        scores는 (B, T, T)이며 행은 Query, 열은 Key 위치입니다.
        triu(1)은 자기 자신을 제외한 미래 Key 위치만 True로 만듭니다.
        masked_fill은 그 점수를 -inf로 바꾸고 softmax로 가중치를 계산합니다.
        x + weights @ v는 원래 표현에 문맥을 더하는 잔차 연결입니다.
        output은 이를 (B, T, 8) logits로 변환합니다.

        logits와 (B, T, T) weights를 반환합니다. 학습 손실은 QUERY
        위치에만 적용하지만 다른 위치의 logits도 인과성 검사에 사용합니다.
        """
        length = ids.size(1)
        x = self.token(ids) + self.position(torch.arange(length, device=ids.device))
        # 정규화한 입력에서 Q·K·V를 각각 만듭니다.
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        # 각 query와 key의 내적으로 위치별 유사도를 계산합니다.
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        # 대각선 위쪽이 현재 위치에서 볼 수 없는 미래입니다.
        future = torch.ones(length, length, dtype=torch.bool, device=ids.device).triu(1)
        # 미래 위치를 소프트맥스 전에 가려 해당 가중치를 0으로 만듭니다.
        weights = scores.masked_fill(future, float('-inf')).softmax(-1)
        return self.output(x + weights @ v), weights


@torch.no_grad()
def evaluate(model, batch):
    """QUERY 답의 정확도와 첫 위치의 어텐션 가중치를 측정합니다.

    logits.flatten(0, 1)은 (B, T, 8)을 (B*T, 8)로 바꿉니다.
    y.flatten()은 (B*T,) 라벨이며 -100인 출력 위치는 손실에서 제외됩니다.
    accuracy는 마지막 위치가 첫 입력 기호를 맞혔는지 계산합니다.
    first_mass는 weights[:, -1, 0], 즉 QUERY가 첫 위치에 준 가중치의 평균입니다.

    반환 지표는 기울기가 없는 Python 숫자입니다.
    """
    model.eval()
    x, y = batch
    logits, weights = model(x)
    return {'loss': F.cross_entropy(logits.flatten(0, 1), y.flatten()).item(),
            'accuracy': (logits[:, -1].argmax(-1) == y[:, -1]).float().mean().item(),
            'first_mass': weights[:, -1, 0].mean().item()}


def main():
    """첫 기호 복사 모델을 학습하고 인과적 정보 범위를 검사합니다.

    --length는 학습 시퀀스 길이이며 위치 임베딩 범위인 2~12만 허용합니다.
    검증 표본을 고정하고 매 단계 새로운 학습 표본 128개를 생성합니다.
    logits의 배치·위치 축을 합친 뒤 -100이 아닌 라벨로 손실을 계산합니다.
    zero_grad, backward, 클리핑, AdamW 갱신 후 주기적으로 검증합니다.

    마지막 두 입력을 바꿔도 그 이전 logits가 유지되는지 검사합니다.
    미래 위치의 가중치가 0인지, 표본을 단독 또는 배치로 계산해도
    결과가 같은지도 확인합니다. 이 검사는 정답 정확도와 별개입니다.
    평가 이력과 예측 예시를 metrics.json, 모델 상태를 model.pt에 저장합니다.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=700)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--length', type=int, default=8)
    parser.add_argument('--out', default='runs/week03')
    args = parser.parse_args()
    if not 2 <= args.length <= 12:
        parser.error('length must be between 2 and 12')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    train_rng = torch.Generator().manual_seed(args.seed + 1)
    validation = make_batch(2048, torch.Generator().manual_seed(args.seed + 1000), args.length)
    model = CausalCopy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    history = [{'step': 0, **evaluate(model, validation)}]
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        x, y = make_batch(128, train_rng, args.length)
        logits, _ = model(x)
        # 정답과 logits를 비교합니다. 소프트맥스는 손실 함수 내부에서 처리합니다.
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
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
        x = validation[0][:4].clone()
        reference, weights = model(x)
        changed = x.clone()
        # 미래 토큰만 바꾸어 앞쪽 출력이 유지되는지 검사합니다.
        changed[:, -2:] = (changed[:, -2:] + 1) % 9
        perturbed, _ = model(changed)
        assert torch.allclose(reference[:, :-2], perturbed[:, :-2], atol=1e-5)
        assert torch.count_nonzero(weights.triu(1)) == 0
        fixed = model(x[:1])[0]
        longer_batch = model(x)[0][:1]
        assert torch.allclose(fixed, longer_batch, atol=1e-5)
    report = {'torch': torch.__version__, 'seed': args.seed, 'length': args.length,
              'seconds': time.perf_counter() - started, 'history': history,
              'sample_input': x[0].tolist(),
              'sample_prediction': reference[0, -1].argmax().item()}
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    torch.save(model.state_dict(), output / 'model.pt')
    print('causality and batch independence: passed')


if __name__ == '__main__':
    main()
```

<a id="section-12"></a>
## 학습 결과와 마스크 검사

기본 설정은 시드 43, 700단계 학습입니다.

| 검증 지표 | 학습 전 | 마지막 단계 |
|---|---:|---:|
| 교차 엔트로피 | 2.61565 | 0.00022 |
| 정확도 | 12.55% | 100.00% |
| 첫 위치의 평균 가중치 | 0.14828 | 0.99999 |

![검증 손실과 정확도 그래프](https://teimakes.github.io/attention-course/week03-learning.svg)

| 시드 | 마지막 단계의 검증 정확도 |
|---|---:|
| 43 | 100.00% |
| 53 | 100.00% |
| 63 | 100.00% |

[측정 기록 보기](https://teimakes.github.io/attention-course/week03-metrics.json)


`accuracy`는 QUERY 위치에서 첫 번째 숫자를 맞힌 비율입니다. 정답 종류가 여덟 개이므로 균등하게 임의 선택하면 기대 정확도는 12.5%입니다. `first_mass`는 마지막 위치가 첫 번째 위치에 부여한 평균 가중치입니다.

코드는 마지막 두 입력을 변경한 뒤 그보다 앞 위치의 logits를 비교합니다. 마스크가 올바르다면 앞쪽 출력은 바뀌지 않아야 합니다. 또 가중치 행렬의 대각선 위가 모두 0인지 검사합니다. 이 두 검사는 모델의 예측 정확도와 별도로 수행해야 합니다. 정답을 잘 맞히더라도 미래 정보를 참조했다면 올바른 자기회귀 모델이 아닙니다.

<a id="section-13"></a>
## 직접 확인해 보기

1. `triu(1)`의 방향을 바꾼 뒤 미래 입력 변경 검사가 어떻게 되는지 확인합니다.
2. 위치 임베딩을 제거하고 처음부터 다시 학습합니다. 마스크 구조 자체도 위치에 관한 정보를 제공할 수 있다는 점을 함께 고려해 결과를 해석합니다.
3. 정답을 첫 번째 숫자 대신 두 번째 숫자로 바꿉니다. 정답뿐 아니라 가중치 분석 위치도 함께 수정합니다.

다음 글에서는 여러 헤드가 서로 다른 가중치를 계산하도록 확장합니다.

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
