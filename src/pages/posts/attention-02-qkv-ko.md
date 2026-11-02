---
layout: '../../layouts/Post.astro'
title: '어텐션 강의 2주차: Q·K·V를 학습하는 정보 검색 모델'
description: '한 주 강의·실습·과제로 배우는 어텐션. 학습 가능한 전체 PyTorch 코드와 실제 측정값, 코드 줄별 해설을 제공합니다.'
date: '2026-11-03'
tags: ['attention', 'transformer', 'pytorch', 'deep-learning']
---

> **대학 수준 어텐션 실습 강의 · 2/5**  
> 이전의 짧은 계산 예제를 학습 가능한 실습과 이론 강의로 전면 개정했습니다. 강의·실습 3시간과 별도 과제를 기준으로 구성했습니다.

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

1. [이번 주의 질문: 같은 자료에서도 질문에 따라 읽을 곳을 바꿀 수 있을까?](#section-1)
2. [1. 데이터와 정답을 먼저 명세하기](#section-2)
3. [2. Q·K·V를 분리하면 무엇이 달라질까?](#section-3)
4. [3. 행렬 연산을 표본 하나에서 배치로 확장하기](#section-4)
5. [4. 왜 sqrt(dk)로 나누는가?](#section-5)
6. [5. 검색은 어떻게 학습되는가?](#section-6)
7. [6. 전체 PyTorch 코드](#section-7)
8. [7. 코드 줄별 해설](#section-8)
9. [8. 실제 측정값과 검색 실패 분석](#section-9)
10. [9. 위치 순서를 바꿔도 결과가 같은 이유](#section-10)
11. [10. 실습 확장과 과제](#section-11)

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
        q = self.wq(self.key_embedding(query)).unsqueeze(1)
        k = self.wk(self.key_embedding(keys))
        v = self.wv(self.value_embedding(values))
        scores = q @ k.transpose(-2, -1)
        if self.scale:
            scores = scores / math.sqrt(k.size(-1))
        weights = scores.softmax(-1)
        if uniform:
            weights = torch.ones_like(weights) / keys.size(1)
        context = (weights @ v).squeeze(1)
        return self.readout(context), weights.squeeze(1)


@torch.no_grad()
def evaluate(model, batch, uniform=False):
    model.eval()
    keys, values, query, target, chosen = batch
    logits, weights = model(keys, values, query, uniform)
    return {
        'loss': F.cross_entropy(logits, target).item(),
        'accuracy': (logits.argmax(-1) == target).float().mean().item(),
        'retrieval_accuracy': (weights.argmax(-1) == chosen).float().mean().item(),
    }


def main():
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
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

<a id="section-8"></a>
## 7. 코드 줄별 해설

- **L1** `"""Week 2: train Q/K/V projections on associative retrieval."""` — 이 실행 파일이 다루는 주제를 설명하는 모듈 문서 문자열입니다.
- **L2** `import argparse` — 명령행 옵션을 읽을 표준 라이브러리입니다.
- **L3** `import json` — 학습 기록을 사람이 다시 읽을 수 있는 JSON으로 저장합니다.
- **L4** `import math` — 어텐션 점수의 제곱근 스케일링에 사용합니다.
- **L5** `import time` — 학습 구간의 경과 시간을 측정합니다.
- **L6** `from pathlib import Path` — 출력 폴더와 저장 파일 경로를 다룹니다.
- **L7** `import torch` — 텐서 연산과 자동 미분을 제공하는 PyTorch를 불러옵니다.
- **L8** `from torch import nn` — 학습 가능한 계층과 Module을 nn이라는 이름으로 사용합니다.
- **L9** `from torch.nn import functional as F` — 손실 함수와 one-hot 등 상태 없는 텐서 연산을 F라는 이름으로 사용합니다.
- **L12** `def make_batch(size, generator, slots=5, key_count=12, classes=6):` — 표본을 생성하는 함수입니다. generator를 외부에서 받아 학습·검증 난수 흐름을 분리합니다.
- **L13** `keys = torch.rand(size, key_count, generator=generator).argsort(-1)[:, :slots]` — 이름별 무작위 점수를 정렬하고 일부를 골라 표본 내 이름 중복을 피합니다.
- **L14** `values = torch.randint(classes, (size, slots), generator=generator)` — 지정 범위의 정수 기호·값·선택 위치 또는 학습 행 인덱스를 무작위로 뽑습니다.
- **L15** `chosen = torch.randint(slots, (size,), generator=generator)` — 지정 범위의 정수 기호·값·선택 위치 또는 학습 행 인덱스를 무작위로 뽑습니다.
- **L16** `rows = torch.arange(size)` — 배치 행 또는 시퀀스 위치의 인덱스를 만들어 같은 표본 안에서 올바른 값을 선택합니다.
- **L17** `return keys, values, keys[rows, chosen], values[rows, chosen], chosen` — 모델 입력과 정답, 분석용 정답 위치를 반환합니다. 위치 정보는 해당 실습의 학습 손실에 직접 넣지 않습니다.
- **L20** `class Retriever(nn.Module):` — `Retriever` 모듈을 정의합니다. 계층의 파라미터 등록과 학습·평가 상태 관리를 PyTorch에 맡깁니다.
- **L21** `def __init__(self, scale=True):` — 학습 가능한 하위 계층을 등록합니다. nn.Module의 속성으로 저장해야 parameters와 state_dict에 포함됩니다.
- **L22** `super().__init__()` — 부모 nn.Module을 초기화해 하위 계층과 파라미터 등록을 사용할 준비를 합니다.
- **L23** `self.scale = scale` — 점수를 sqrt(dk)로 나눌지 실행 옵션을 모델 설정으로 보관합니다.
- **L24** `self.key_embedding = nn.Embedding(12, 32)` — 범주 ID를 학습 가능한 밀집 벡터로 바꾸는 조회표를 등록합니다. 위치 임베딩도 같은 원리로 학습됩니다.
- **L25** `self.value_embedding = nn.Embedding(6, 32)` — 범주 ID를 학습 가능한 밀집 벡터로 바꾸는 조회표를 등록합니다. 위치 임베딩도 같은 원리로 학습됩니다.
- **L26** `self.wq = nn.Linear(32, 32, bias=False)` — 지정한 입력·출력 차원의 학습 가능한 선형 변환을 등록합니다. bias=False인 계층에는 편향을 두지 않습니다.
- **L27** `self.wk = nn.Linear(32, 32, bias=False)` — 지정한 입력·출력 차원의 학습 가능한 선형 변환을 등록합니다. bias=False인 계층에는 편향을 두지 않습니다.
- **L28** `self.wv = nn.Linear(32, 32, bias=False)` — 지정한 입력·출력 차원의 학습 가능한 선형 변환을 등록합니다. bias=False인 계층에는 편향을 두지 않습니다.
- **L29** `self.readout = nn.Linear(32, 6)` — 지정한 입력·출력 차원의 학습 가능한 선형 변환을 등록합니다. bias=False인 계층에는 편향을 두지 않습니다.
- **L31** `def forward(self, keys, values, query, uniform=False):` — 입력 텐서로부터 출력까지의 순전파를 정의합니다. 학습 시 자동 미분 그래프가 이 연산을 기록합니다.
- **L32** `q = self.wq(self.key_embedding(query)).unsqueeze(1)` — 입력 표현을 Q·K·V로 투영하고 필요하면 헤드 축으로 나눕니다. 셀프 어텐션은 같은 입력, 교차 어텐션은 서로 다른 입력을 사용합니다.
- **L33** `k = self.wk(self.key_embedding(keys))` — 입력 표현을 Q·K·V로 투영하고 필요하면 헤드 축으로 나눕니다. 셀프 어텐션은 같은 입력, 교차 어텐션은 서로 다른 입력을 사용합니다.
- **L34** `v = self.wv(self.value_embedding(values))` — 입력 표현을 Q·K·V로 투영하고 필요하면 헤드 축으로 나눕니다. 셀프 어텐션은 같은 입력, 교차 어텐션은 서로 다른 입력을 사용합니다.
- **L35** `scores = q @ k.transpose(-2, -1)` — query와 key의 내적을 계산합니다. 스케일링이 같은 줄에 있으면 헤드 차원의 제곱근으로 나눕니다.
- **L36** `if self.scale:` — 스케일링을 켠 실행에서만 점수를 sqrt(dk)로 나눕니다.
- **L37** `scores = scores / math.sqrt(k.size(-1))` — query와 key의 내적을 계산합니다. 스케일링이 같은 줄에 있으면 헤드 차원의 제곱근으로 나눕니다.
- **L38** `weights = scores.softmax(-1)` — 각 query가 참고할 key 위치의 가중치를 합이 1이 되도록 계산합니다.
- **L39** `if uniform:` — 학습된 선택 가중치 대신 균등 가중치로 바꾸는 평가 개입입니다. 재학습 기준선과 구분합니다.
- **L40** `weights = torch.ones_like(weights) / keys.size(1)` — 모든 key 위치에 동일한 비율을 부여해 학습된 선택을 제거하는 개입을 만듭니다.
- **L41** `context = (weights @ v).squeeze(1)` — 가중치로 value를 모으거나 헤드 축을 되돌려 결합합니다. 행렬곱의 마지막 두 축을 확인하세요.
- **L42** `return self.readout(context), weights.squeeze(1)` — 예측 logits와 어텐션 가중치를 함께 반환해 성능과 내부 선택을 따로 분석합니다.
- **L46** `def evaluate(model, batch, uniform=False):` — 검증 지표를 계산합니다. no_grad와 eval을 함께 사용해 평가와 학습을 구분합니다.
- **L47** `model.eval()` — 모델을 평가 모드로 설정합니다. 이 호출만으로 자동 미분이 꺼지지는 않습니다.
- **L48** `keys, values, query, target, chosen = batch` — 준비된 표본 묶음을 입력·정답·분석용 선택 위치로 풀어 각 계산에 전달합니다.
- **L49** `logits, weights = model(keys, values, query, uniform)` — 모델의 순전파 결과에서 점수 또는 가중치를 받아 예측과 내부 선택을 분석합니다.
- **L50–54** `return {` — 손실과 정확도 등 평가 지표를 Python 숫자 사전으로 반환합니다. item은 텐서의 스칼라를 꺼냅니다.
- **L57** `def main():` — 옵션 파싱부터 데이터 준비, 학습, 평가와 저장까지 실행 순서를 연결합니다.
- **L58** `parser = argparse.ArgumentParser()` — 명령행 실행 설정을 구성하거나 실제 입력 옵션을 해석합니다.
- **L59** `parser.add_argument('--steps', type=int, default=700)` — 실행 옵션의 이름·자료형·기본값을 등록합니다. --out을 바꾸면 이전 실험 기록을 보존할 수 있습니다.
- **L60** `parser.add_argument('--seed', type=int, default=42)` — 실행 옵션의 이름·자료형·기본값을 등록합니다. --out을 바꾸면 이전 실험 기록을 보존할 수 있습니다.
- **L61** `parser.add_argument('--no-scale', action='store_true')` — 실행 옵션의 이름·자료형·기본값을 등록합니다. --out을 바꾸면 이전 실험 기록을 보존할 수 있습니다.
- **L62** `parser.add_argument('--out', default='runs/week02')` — 실행 옵션의 이름·자료형·기본값을 등록합니다. --out을 바꾸면 이전 실험 기록을 보존할 수 있습니다.
- **L63** `args = parser.parse_args()` — 명령행 실행 설정을 구성하거나 실제 입력 옵션을 해석합니다.
- **L64** `torch.set_num_threads(1)` — 작은 CPU 실습의 스레드 과잉 비용을 줄이고 실행 환경을 단순화합니다.
- **L65** `torch.manual_seed(args.seed)` — 모델 초기화와 기본 난수의 시드를 설정합니다. 버전·장치가 달라지면 완전한 수치 일치까지 보장하지는 않습니다.
- **L66** `train_rng = torch.Generator().manual_seed(args.seed + 1)` — 독립적인 난수 생성기를 만들고 시드를 고정해 학습과 평가 데이터 흐름을 통제합니다.
- **L67** `validation = make_batch(2048, torch.Generator().manual_seed(args.seed + 1000))` — 고정 평가 표본 또는 이번 step의 학습 표본을 준비합니다. 학습·검증·테스트의 역할을 구분하세요.
- **L68** `model = Retriever(scale=not args.no_scale)` — 지정한 구성으로 학습 가능한 모델을 만듭니다. 저장 상태를 읽기 전에는 초기 파라미터입니다.
- **L69** `optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)` — 등록된 학습 파라미터를 AdamW로 최적화하도록 설정합니다. 학습률과 감쇠값은 코드에 명시합니다.
- **L70** `history = [{'step': 0, **evaluate(model, validation)}]` — 업데이트 전의 검증 지표부터 기록해 학습 효과를 초기 상태와 비교할 수 있게 합니다.
- **L71** `started = time.perf_counter()` — 학습 및 최종 진단 구간의 경과 시간 측정을 시작합니다.
- **L72** `for step in range(1, args.steps + 1):` — 새 학습 배치를 사용해 지정한 횟수만큼 파라미터 업데이트를 반복합니다.
- **L73** `model.train()` — 모델을 학습 모드로 되돌립니다. 검증 직후에도 이 호출이 필요합니다.
- **L74** `keys, values, query, target, _ = make_batch(128, train_rng)` — 고정 평가 표본 또는 이번 step의 학습 표본을 준비합니다. 학습·검증·테스트의 역할을 구분하세요.
- **L75** `logits, _ = model(keys, values, query)` — 모델의 순전파 결과에서 점수 또는 가중치를 받아 예측과 내부 선택을 분석합니다.
- **L76** `loss = F.cross_entropy(logits, target)` — 소프트맥스 전 logits와 정수 정답으로 평균 교차 엔트로피를 계산합니다. -100 라벨은 기본적으로 무시됩니다.
- **L77** `optimizer.zero_grad(set_to_none=True)` — 이전 step의 기울기를 비워 의도하지 않은 누적을 방지합니다.
- **L78** `loss.backward()` — 손실에서 점수·투영·임베딩까지 연쇄 법칙으로 기울기를 계산합니다.
- **L79** `nn.utils.clip_grad_norm_(model.parameters(), 1.0)` — 전체 파라미터 기울기의 노름을 제한해 큰 업데이트를 완화합니다.
- **L80** `optimizer.step()` — 계산된 기울기와 AdamW의 상태를 사용해 파라미터를 갱신합니다.
- **L81** `if step % 100 == 0 or step == args.steps:` — 정해진 간격과 마지막 단계에서 검증·로그를 실행합니다. 학습 step과 epoch는 다릅니다.
- **L82** `metrics = {'step': step, 'train_loss': loss.item(), **evaluate(model, validation)}` — 학습 단계와 손실, 고정 검증 데이터의 지표를 묶습니다. 검증 후 다음 step에서 학습 모드로 복귀합니다.
- **L83** `history.append(metrics)` — 이번 검증 지표를 순서대로 기록해 학습 곡선과 초기·최종 비교에 사용합니다.
- **L84** `print(json.dumps(metrics))` — 현재 실행의 지표나 진단 결과를 터미널에 출력합니다. JSON 로그는 후처리하기 쉽습니다.
- **L85** `model.eval()` — 모델을 평가 모드로 설정합니다. 이 호출만으로 자동 미분이 꺼지지는 않습니다.
- **L86** `with torch.no_grad():` — 이 블록에서는 기울기 기록을 꺼서 검증과 구조 검사에 필요한 계산만 수행합니다.
- **L87** `keys, values, query, _, _ = validation` — 준비된 표본 묶음을 입력·정답·분석용 선택 위치로 풀어 각 계산에 전달합니다.
- **L88** `before, weights = model(keys[:4], values[:4], query[:4])` — 모델의 순전파 결과에서 점수 또는 가중치를 받아 예측과 내부 선택을 분석합니다.
- **L89** `permutation = torch.tensor([4, 2, 0, 1, 3])` — 카드 순서만 바꾸기 위한 고정 순열을 정의합니다.
- **L90** `after, _ = model(keys[:4, permutation], values[:4, permutation], query[:4])` — 모델의 순전파 결과에서 점수 또는 가중치를 받아 예측과 내부 선택을 분석합니다.
- **L91** `assert torch.allclose(before, after, atol=1e-5)` — 카드 이름과 값을 함께 순열 변경해도 출력이 같은지 검사합니다.
- **L92** `assert torch.allclose(weights.sum(-1), torch.ones(4), atol=1e-6)` — key 위치 축의 가중치 합이 1인지 검사합니다.
- **L93–95** `report = {'torch': torch.__version__, 'seed': args.seed, 'scale': model.scale,` — 환경·실험 설정·학습 이력·진단 또는 테스트 결과를 하나의 재현 가능한 기록으로 묶습니다.
- **L96** `output = Path(args.out)` — 사용자가 지정한 결과 저장 경로를 Path 객체로 만듭니다.
- **L97** `output.mkdir(parents=True, exist_ok=True)` — 기록과 체크포인트를 저장할 출력 폴더를 만듭니다.
- **L98** `(output / 'metrics.json').write_text(json.dumps(report, indent=2))` — 실행 설정과 측정 기록을 JSON 파일로 저장합니다. 글의 그래프는 이 숫자에서 생성됩니다.
- **L99** `torch.save(model.state_dict(), output / 'model.pt')` — 모델의 학습된 상태를 파일에 저장합니다. 5주차에서는 어휘와 선택 step도 함께 보관합니다.
- **L100** `print('permutation invariance: passed')` — 현재 실행의 지표나 진단 결과를 터미널에 출력합니다. JSON 로그는 후처리하기 쉽습니다.
- **L103** `if __name__ == '__main__':` — 파일을 직접 실행할 때 main을 호출하고, 다른 코드에서 import할 때는 학습을 시작하지 않도록 합니다.
- **L104** `main()` — 명령행에서 지정한 설정으로 전체 학습 실험을 실행합니다.

<a id="section-9"></a>
## 8. 실제 측정값과 검색 실패 분석

측정 환경은 Python 3.12.14, PyTorch 2.15.0+cpu, CPU, 스레드 1개입니다. 기본 시드는 42이고 마지막 기록은 700 step입니다. 아래 숫자와 그래프는 실제 실행에서 저장한 값입니다. 시간은 기기와 병렬 작업에 따라 달라지는 참고값입니다.

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
## 9. 위치 순서를 바꿔도 결과가 같은 이유

카드들을 같은 순열로 섞으면 K와 V의 행이 함께 바뀝니다. 점수의 열과 가중치의 열도 같은 방식으로 바뀌므로 최종 가중합은 그대로입니다. 이 구조에는 위치 임베딩이 없고 카드의 순서 자체가 정답과 무관하기 때문입니다.

코드 마지막의 순열 검사는 이런 성질을 실제 계산에서 확인합니다. K만 섞고 V를 그대로 두면 이름과 값의 대응이 깨져 다른 입력이 됩니다. 두 종류의 순서 변경을 구분해야 합니다. 다음 주처럼 위치가 의미를 가지는 문제에서는 이 불변성이 오히려 한계가 됩니다.

<a id="section-11"></a>
## 10. 실습 확장과 과제

1. 기본 실행과 `--no-scale --out runs/week02-no-scale`를 같은 시드 세 개로 비교합니다. 최종 정확도뿐 아니라 첫 100단계 손실과 가중치 엔트로피도 측정하세요.
2. 카드 수를 5에서 10으로 늘립니다. 데이터 함수와 순열 검사 길이를 함께 수정해야 합니다. 검색 후보 증가가 학습에 미치는 영향을 설명하세요.
3. 질문을 검증 카드에 없는 이름으로 바꿉니다. 현재 모델에는 ‘없음’ 클래스를 학습하지 않았습니다. 확률이 높게 출력되더라도 답이 존재함을 보장하지 않는 이유를 설명하세요.
4. Q와 K 투영을 공유하도록 바꾼 모델을 처음부터 학습하고 파라미터 수와 성능을 비교하세요.

과제는 이론 25점, Q·K·V 모양과 배치 구현 30점, 세 시드 비교 30점, 일반화 범위와 실패 해석 15점입니다. 보고서에는 최소 다섯 실패 사례의 카드·질문·정답·예측·가중치를 함께 제시합니다. 실패가 없다면 카드 수나 학습 단계를 바꾸어 경계 조건을 찾아보세요.

## 읽을 자료

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 특히 3.2~3.5절. 구조와 수식의 원 출처입니다.
- [PyTorch SDPA 공식 문서](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html): 마스크와 스케일링의 구현 의미를 확인합니다.
- [PyTorch CrossEntropyLoss 공식 문서](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html): logits, 정수 타깃과 무시 인덱스를 확인합니다.

이 글의 합성 데이터·실습 코드는 강의를 위해 새롭게 작성했습니다.
