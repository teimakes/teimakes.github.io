---
layout: '../../layouts/Post.astro'
title: '어텐션 강의 3주차: 인과적 셀프 어텐션과 정보 누출'
description: '한 주 강의·실습·과제로 배우는 어텐션. 학습 가능한 전체 PyTorch 코드와 실제 측정값, 코드 줄별 해설을 제공합니다.'
date: '2026-11-03'
tags: ['attention', 'transformer', 'pytorch', 'deep-learning']
---

> **대학 수준 어텐션 실습 강의 · 3/5**  
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

1. [이번 주의 질문: 병렬로 학습하면서도 미래를 못 보게 할 수 있을까?](#section-1)
2. [1. 자기회귀 언어 모델의 정보 범위](#section-2)
3. [2. 지연 복사 과제 명세](#section-3)
4. [3. 셀프 어텐션과 위치 임베딩](#section-4)
5. [4. 마스크의 행과 열을 확실히 구분하기](#section-5)
6. [5. 전체 PyTorch 코드](#section-6)
7. [6. 코드 줄별 해설](#section-7)
8. [7. 학습 결과와 인과성 검증](#section-8)
9. [8. 길이와 평가 범위](#section-9)
10. [9. 과제: 정보 누출을 직접 찾아내기](#section-10)

<a id="section-1"></a>
## 이번 주의 질문: 병렬로 학습하면서도 미래를 못 보게 할 수 있을까?

다음 토큰을 예측하는 모델이 학습 중 정답 토큰을 미리 볼 수 있다면 무엇을 배울까요? 앞의 문맥에서 규칙을 찾는 대신 답을 복사하는 쉬운 길을 택할 수 있습니다. 이런 정보 누출을 막기 위해 인과적 셀프 어텐션은 현재 위치보다 뒤에 있는 입력을 가립니다.

이번 주에는 Q·K·V를 같은 시퀀스에서 만들고 **오래전에 나온 정보를 나중의 질문 위치에서 복사하는 모델**을 학습합니다. 가중치 표에 0이 보이는지뿐 아니라, 미래 입력을 바꿔도 과거 출력이 실제로 불변인지 검사합니다.

### 한 주의 구성과 학습 목표

강의 90분: 자기회귀 확률분해 25분, 마스크와 위치 정보 35분, 손실 위치와 정보 누출 30분. 실습 90분: 학습·미래 변경 검사·길이 변경 분석. 과제 2~3시간: 마스크 방향, 위치 임베딩 제거, 학습 길이 변경에 대한 실험 보고서.

학습 후에는 ‘자기 자신을 볼 수 있는가’, ‘정답을 어디에 놓는가’, ‘마스크를 언제 적용하는가’를 코드로 설명할 수 있어야 합니다. 학습 중 병렬 계산과 생성 중 순차 계산의 차이도 구분합니다.

<a id="section-2"></a>
## 1. 자기회귀 언어 모델의 정보 범위

```text
P(x₁,…,xT) = Πₜ P(xₜ | x₁,…,xₜ₋₁)
입력 : [A, B, C]
정답 : [B, C, D]
```

입력 위치 0의 출력은 A까지 보고 B를 예측합니다. 위치 1의 출력은 A와 B를 보고 C를 예측합니다. 그래서 입력 위치는 자기 자신을 볼 수 있지만 다음 위치를 볼 수는 없습니다. 학습 데이터의 입력과 정답을 한 칸 어긋나게 구성하는 것이 핵심입니다.

이번 실습은 모든 위치의 다음 토큰을 학습하기 전에, **질문 뒤에 나오는 한 개의 정답**에만 손실을 적용합니다. 따라서 전체 언어 모델 학습과 동일하지는 않으며, 5주차에서 모든 위치의 다음 토큰 손실로 확장합니다.

<a id="section-3"></a>
## 2. 지연 복사 과제 명세

입력 길이는 기본 8입니다. 처음 일곱 위치에는 0~7의 무작위 기호가 있고 마지막 위치에는 QUERY를 나타내는 ID 8이 있습니다. 정답은 입력의 첫 기호입니다.

```text
입력: [3, 6, 1, 3, 0, 7, 2, QUERY]
QUERY 뒤에 생성할 정답: 3
손실 라벨: [ignore, ignore, …, ignore, 3]
```

중간 기호들은 정답과 무관합니다. 정답을 알려면 QUERY 위치에서 첫 위치의 정보를 가져와야 합니다. 입력에 같은 기호가 우연히 반복되더라도 정답 정의는 ‘첫 위치’입니다. 이 점에서 카드 순서에 불변이었던 2주차와 다릅니다.

라벨의 `-100`은 PyTorch 교차 엔트로피의 기본 무시 인덱스입니다. 라벨을 무시한다는 것은 그 위치의 **출력 손실을 계산하지 않는 것**입니다. 해당 입력 위치가 어텐션에서 사라진다는 뜻은 아닙니다. 첫 위치는 자기 위치에 손실이 없어도 QUERY의 정답을 맞히는 경로를 통해 기울기를 받을 수 있습니다.

<a id="section-4"></a>
## 3. 셀프 어텐션과 위치 임베딩

각 토큰의 입력은 토큰 임베딩과 위치 임베딩을 더해 만듭니다. `nn.Embedding(9,32)`가 기호의 표현을, `nn.Embedding(12,32)`가 최대 12개 위치의 표현을 저장합니다. 첫 위치를 찾아야 하므로 어느 위치인지를 구분하는 신호가 필요합니다.

`qkv` 계층은 한 번에 차원 96의 벡터를 만들고 이를 32차원씩 세 조각으로 나눕니다. 서로 다른 투영 세 개를 하나의 큰 선형 계층으로 합친 구현입니다. 이번에는 배치 `(B,T,D)`의 마지막 차원을 변환하고, Q·K·V는 모두 같은 입력에서 나옵니다.

```text
X = token(ids) + position(0,…,T−1)
[Q,K,V] = Linear(LayerNorm(X))
S = QKᵀ / sqrt(D)
A = softmax(S + M)
logits = Linear(X + AV)
```

단일 헤드와 잔차 연결, 입력 정규화를 사용하지만 별도의 FFN은 아직 넣지 않았습니다. 단계별로 무엇이 추가되는지 보이게 하기 위한 구성입니다.

<a id="section-5"></a>
## 4. 마스크의 행과 열을 확실히 구분하기

점수 행렬의 행은 query 위치, 열은 참고할 key 위치입니다. `M[i,j]`는 j가 i보다 뒤라면 음의 무한대이고 나머지는 0입니다. 코드에서는 `True`가 금지된 위치임을 뜻하는 `future`를 만든 뒤 `masked_fill`로 점수를 바꿉니다.

소프트맥스 **이전**에 마스크를 적용하면 금지 점수의 지수가 0이 됩니다. 소프트맥스 후에 단순히 0을 곱하면 행 합이 1이 아니게 될 수 있고, 금지 위치가 정규화 분모에 이미 영향을 주었을 수 있습니다. 이 순서를 바꾸지 마세요.

주의할 API 차이도 있습니다. 이 코드의 `masked_fill`에서 True는 금지이지만, PyTorch `scaled_dot_product_attention`에 직접 넘기는 boolean `attn_mask`의 True는 허용을 의미합니다. 다른 API로 옮길 때 마스크를 그대로 복사하지 말고 공식 문서의 의미를 확인해야 합니다.

<a id="section-6"></a>
## 5. 전체 PyTorch 코드

전체 소스는 [week03.py](/attention-course/week03.py)로도 받을 수 있습니다. 아래 코드는 같은 파일의 전체 내용입니다.

```bash
python week03.py
```

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
    inputs = torch.randint(8, (size, length), generator=generator)
    inputs[:, -1] = 8  # QUERY marker
    labels = torch.full_like(inputs, -100)
    labels[:, -1] = inputs[:, 0]  # answer after QUERY copies first symbol
    return inputs, labels


class CausalCopy(nn.Module):
    def __init__(self, context=12, dim=32):
        super().__init__()
        self.token = nn.Embedding(9, dim)
        self.position = nn.Embedding(context, dim)
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, 8)

    def forward(self, ids):
        length = ids.size(1)
        x = self.token(ids) + self.position(torch.arange(length, device=ids.device))
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        future = torch.ones(length, length, dtype=torch.bool, device=ids.device).triu(1)
        weights = scores.masked_fill(future, float('-inf')).softmax(-1)
        return self.output(x + weights @ v), weights


@torch.no_grad()
def evaluate(model, batch):
    model.eval()
    x, y = batch
    logits, weights = model(x)
    return {'loss': F.cross_entropy(logits.flatten(0, 1), y.flatten()).item(),
            'accuracy': (logits[:, -1].argmax(-1) == y[:, -1]).float().mean().item(),
            'first_mass': weights[:, -1, 0].mean().item()}


def main():
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
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
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
        x = validation[0][:4].clone()
        reference, weights = model(x)
        changed = x.clone()
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

<a id="section-7"></a>
## 6. 코드 줄별 해설

- **L1** `"""Week 3: train causal self-attention for delayed copy."""` — 이 실행 파일이 다루는 주제를 설명하는 모듈 문서 문자열입니다.
- **L2** `import argparse` — 명령행 옵션을 읽을 표준 라이브러리입니다.
- **L3** `import json` — 학습 기록을 사람이 다시 읽을 수 있는 JSON으로 저장합니다.
- **L4** `import math` — 어텐션 점수의 제곱근 스케일링에 사용합니다.
- **L5** `import time` — 학습 구간의 경과 시간을 측정합니다.
- **L6** `from pathlib import Path` — 출력 폴더와 저장 파일 경로를 다룹니다.
- **L7** `import torch` — 텐서 연산과 자동 미분을 제공하는 PyTorch를 불러옵니다.
- **L8** `from torch import nn` — 학습 가능한 계층과 Module을 nn이라는 이름으로 사용합니다.
- **L9** `from torch.nn import functional as F` — 손실 함수와 one-hot 등 상태 없는 텐서 연산을 F라는 이름으로 사용합니다.
- **L12** `def make_batch(size, generator, length=8):` — 표본을 생성하는 함수입니다. generator를 외부에서 받아 학습·검증 난수 흐름을 분리합니다.
- **L13** `inputs = torch.randint(8, (size, length), generator=generator)` — 지정 범위의 정수 기호·값·선택 위치 또는 학습 행 인덱스를 무작위로 뽑습니다.
- **L14** `inputs[:, -1] = 8  # QUERY marker` — 마지막 입력을 QUERY라는 전용 기호 ID로 바꿉니다.
- **L15** `labels = torch.full_like(inputs, -100)` — 모든 출력 위치를 무시 인덱스로 초기화한 뒤 실제 감독할 위치만 정답으로 바꿀 준비를 합니다.
- **L16** `labels[:, -1] = inputs[:, 0]  # answer after QUERY copies first symbol` — QUERY 위치의 다음 토큰 정답을 첫 입력 기호로 설정합니다. 다른 위치의 출력 손실은 무시됩니다.
- **L17** `return inputs, labels` — 모델 입력과 정답, 분석용 정답 위치를 반환합니다. 위치 정보는 해당 실습의 학습 손실에 직접 넣지 않습니다.
- **L20** `class CausalCopy(nn.Module):` — `CausalCopy` 모듈을 정의합니다. 계층의 파라미터 등록과 학습·평가 상태 관리를 PyTorch에 맡깁니다.
- **L21** `def __init__(self, context=12, dim=32):` — 학습 가능한 하위 계층을 등록합니다. nn.Module의 속성으로 저장해야 parameters와 state_dict에 포함됩니다.
- **L22** `super().__init__()` — 부모 nn.Module을 초기화해 하위 계층과 파라미터 등록을 사용할 준비를 합니다.
- **L23** `self.token = nn.Embedding(9, dim)` — 범주 ID를 학습 가능한 밀집 벡터로 바꾸는 조회표를 등록합니다. 위치 임베딩도 같은 원리로 학습됩니다.
- **L24** `self.position = nn.Embedding(context, dim)` — 범주 ID를 학습 가능한 밀집 벡터로 바꾸는 조회표를 등록합니다. 위치 임베딩도 같은 원리로 학습됩니다.
- **L25** `self.norm = nn.LayerNorm(dim)` — 각 토큰의 특성 축을 정규화하는 학습 가능한 LayerNorm 계층을 등록합니다.
- **L26** `self.qkv = nn.Linear(dim, 3 * dim)` — 지정한 입력·출력 차원의 학습 가능한 선형 변환을 등록합니다. bias=False인 계층에는 편향을 두지 않습니다.
- **L27** `self.output = nn.Linear(dim, 8)` — 지정한 입력·출력 차원의 학습 가능한 선형 변환을 등록합니다. bias=False인 계층에는 편향을 두지 않습니다.
- **L29** `def forward(self, ids):` — 입력 텐서로부터 출력까지의 순전파를 정의합니다. 학습 시 자동 미분 그래프가 이 연산을 기록합니다.
- **L30** `length = ids.size(1)` — 현재 입력의 배치·길이·특성 차원을 읽거나 위치 인덱스를 생성합니다.
- **L31** `x = self.token(ids) + self.position(torch.arange(length, device=ids.device))` — 배치 행 또는 시퀀스 위치의 인덱스를 만들어 같은 표본 안에서 올바른 값을 선택합니다.
- **L32** `q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)` — 입력 표현을 Q·K·V로 투영하고 필요하면 헤드 축으로 나눕니다. 셀프 어텐션은 같은 입력, 교차 어텐션은 서로 다른 입력을 사용합니다.
- **L33** `scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))` — query와 key의 내적을 계산합니다. 스케일링이 같은 줄에 있으면 헤드 차원의 제곱근으로 나눕니다.
- **L34** `future = torch.ones(length, length, dtype=torch.bool, device=ids.device).triu(1)` — 대각선을 제외한 위쪽 삼각형을 True로 만들어 현재 query보다 뒤쪽 key를 표시합니다.
- **L35** `weights = scores.masked_fill(future, float('-inf')).softmax(-1)` — 미래 위치 점수를 음의 무한대로 바꾼 다음 key 축 소프트맥스로 정규화합니다.
- **L36** `return self.output(x + weights @ v), weights` — 예측 logits와 어텐션 가중치를 함께 반환해 성능과 내부 선택을 따로 분석합니다.
- **L40** `def evaluate(model, batch):` — 검증 지표를 계산합니다. no_grad와 eval을 함께 사용해 평가와 학습을 구분합니다.
- **L41** `model.eval()` — 모델을 평가 모드로 설정합니다. 이 호출만으로 자동 미분이 꺼지지는 않습니다.
- **L42** `x, y = batch` — 준비된 표본 묶음을 입력·정답·분석용 선택 위치로 풀어 각 계산에 전달합니다.
- **L43** `logits, weights = model(x)` — 모델의 순전파 결과에서 점수 또는 가중치를 받아 예측과 내부 선택을 분석합니다.
- **L44–46** `return {'loss': F.cross_entropy(logits.flatten(0, 1), y.flatten()).item(),` — 손실과 정확도 등 평가 지표를 Python 숫자 사전으로 반환합니다. item은 텐서의 스칼라를 꺼냅니다.
- **L49** `def main():` — 옵션 파싱부터 데이터 준비, 학습, 평가와 저장까지 실행 순서를 연결합니다.
- **L50** `parser = argparse.ArgumentParser()` — 명령행 실행 설정을 구성하거나 실제 입력 옵션을 해석합니다.
- **L51** `parser.add_argument('--steps', type=int, default=700)` — 실행 옵션의 이름·자료형·기본값을 등록합니다. --out을 바꾸면 이전 실험 기록을 보존할 수 있습니다.
- **L52** `parser.add_argument('--seed', type=int, default=43)` — 실행 옵션의 이름·자료형·기본값을 등록합니다. --out을 바꾸면 이전 실험 기록을 보존할 수 있습니다.
- **L53** `parser.add_argument('--length', type=int, default=8)` — 실행 옵션의 이름·자료형·기본값을 등록합니다. --out을 바꾸면 이전 실험 기록을 보존할 수 있습니다.
- **L54** `parser.add_argument('--out', default='runs/week03')` — 실행 옵션의 이름·자료형·기본값을 등록합니다. --out을 바꾸면 이전 실험 기록을 보존할 수 있습니다.
- **L55** `args = parser.parse_args()` — 명령행 실행 설정을 구성하거나 실제 입력 옵션을 해석합니다.
- **L56** `if not 2 <= args.length <= 12:` — 입력 길이·헤드 차원·온도 등의 전제 조건을 검사해 잘못된 실행을 차단합니다.
- **L57** `parser.error('length must be between 2 and 12')` — 지원하지 않는 길이 설정을 사용자에게 알리고 실행을 종료합니다.
- **L58** `torch.set_num_threads(1)` — 작은 CPU 실습의 스레드 과잉 비용을 줄이고 실행 환경을 단순화합니다.
- **L59** `torch.manual_seed(args.seed)` — 모델 초기화와 기본 난수의 시드를 설정합니다. 버전·장치가 달라지면 완전한 수치 일치까지 보장하지는 않습니다.
- **L60** `train_rng = torch.Generator().manual_seed(args.seed + 1)` — 독립적인 난수 생성기를 만들고 시드를 고정해 학습과 평가 데이터 흐름을 통제합니다.
- **L61** `validation = make_batch(2048, torch.Generator().manual_seed(args.seed + 1000), args.length)` — 고정 평가 표본 또는 이번 step의 학습 표본을 준비합니다. 학습·검증·테스트의 역할을 구분하세요.
- **L62** `model = CausalCopy()` — 지정한 구성으로 학습 가능한 모델을 만듭니다. 저장 상태를 읽기 전에는 초기 파라미터입니다.
- **L63** `optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)` — 등록된 학습 파라미터를 AdamW로 최적화하도록 설정합니다. 학습률과 감쇠값은 코드에 명시합니다.
- **L64** `history = [{'step': 0, **evaluate(model, validation)}]` — 업데이트 전의 검증 지표부터 기록해 학습 효과를 초기 상태와 비교할 수 있게 합니다.
- **L65** `started = time.perf_counter()` — 학습 및 최종 진단 구간의 경과 시간 측정을 시작합니다.
- **L66** `for step in range(1, args.steps + 1):` — 새 학습 배치를 사용해 지정한 횟수만큼 파라미터 업데이트를 반복합니다.
- **L67** `model.train()` — 모델을 학습 모드로 되돌립니다. 검증 직후에도 이 호출이 필요합니다.
- **L68** `x, y = make_batch(128, train_rng, args.length)` — 고정 평가 표본 또는 이번 step의 학습 표본을 준비합니다. 학습·검증·테스트의 역할을 구분하세요.
- **L69** `logits, _ = model(x)` — 모델의 순전파 결과에서 점수 또는 가중치를 받아 예측과 내부 선택을 분석합니다.
- **L70** `loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())` — 소프트맥스 전 logits와 정수 정답으로 평균 교차 엔트로피를 계산합니다. -100 라벨은 기본적으로 무시됩니다.
- **L71** `optimizer.zero_grad(set_to_none=True)` — 이전 step의 기울기를 비워 의도하지 않은 누적을 방지합니다.
- **L72** `loss.backward()` — 손실에서 점수·투영·임베딩까지 연쇄 법칙으로 기울기를 계산합니다.
- **L73** `nn.utils.clip_grad_norm_(model.parameters(), 1.0)` — 전체 파라미터 기울기의 노름을 제한해 큰 업데이트를 완화합니다.
- **L74** `optimizer.step()` — 계산된 기울기와 AdamW의 상태를 사용해 파라미터를 갱신합니다.
- **L75** `if step % 100 == 0 or step == args.steps:` — 정해진 간격과 마지막 단계에서 검증·로그를 실행합니다. 학습 step과 epoch는 다릅니다.
- **L76** `metrics = {'step': step, 'train_loss': loss.item(), **evaluate(model, validation)}` — 학습 단계와 손실, 고정 검증 데이터의 지표를 묶습니다. 검증 후 다음 step에서 학습 모드로 복귀합니다.
- **L77** `history.append(metrics)` — 이번 검증 지표를 순서대로 기록해 학습 곡선과 초기·최종 비교에 사용합니다.
- **L78** `print(json.dumps(metrics))` — 현재 실행의 지표나 진단 결과를 터미널에 출력합니다. JSON 로그는 후처리하기 쉽습니다.
- **L79** `model.eval()` — 모델을 평가 모드로 설정합니다. 이 호출만으로 자동 미분이 꺼지지는 않습니다.
- **L80** `with torch.no_grad():` — 이 블록에서는 기울기 기록을 꺼서 검증과 구조 검사에 필요한 계산만 수행합니다.
- **L81** `x = validation[0][:4].clone()` — 고정 검증 입력의 일부를 복사해 원본 데이터에 영향을 주지 않고 구조 검사를 수행합니다.
- **L82** `reference, weights = model(x)` — 모델의 순전파 결과에서 점수 또는 가중치를 받아 예측과 내부 선택을 분석합니다.
- **L83** `changed = x.clone()` — 미래 입력만 바꾼 복사본을 만들어 과거 출력의 불변성을 검사합니다.
- **L84** `changed[:, -2:] = (changed[:, -2:] + 1) % 9` — 미래 입력만 바꾼 복사본을 만들어 과거 출력의 불변성을 검사합니다.
- **L85** `perturbed, _ = model(changed)` — 모델의 순전파 결과에서 점수 또는 가중치를 받아 예측과 내부 선택을 분석합니다.
- **L86** `assert torch.allclose(reference[:, :-2], perturbed[:, :-2], atol=1e-5)` — 미래 입력만 변경했을 때 앞쪽 출력이 변하지 않는지 수치 오차 허용 범위에서 검사합니다.
- **L87** `assert torch.count_nonzero(weights.triu(1)) == 0` — 금지된 미래 위치의 어텐션 가중치가 정확히 0인지 검사합니다.
- **L88** `fixed = model(x[:1])[0]` — 모델의 순전파 결과에서 점수 또는 가중치를 받아 예측과 내부 선택을 분석합니다.
- **L89** `longer_batch = model(x)[0][:1]` — 모델의 순전파 결과에서 점수 또는 가중치를 받아 예측과 내부 선택을 분석합니다.
- **L90** `assert torch.allclose(fixed, longer_batch, atol=1e-5)` — 표본을 단독 계산하거나 배치로 계산해도 결과가 같은지 검사합니다.
- **L91–94** `report = {'torch': torch.__version__, 'seed': args.seed, 'length': args.length,` — 환경·실험 설정·학습 이력·진단 또는 테스트 결과를 하나의 재현 가능한 기록으로 묶습니다.
- **L95** `output = Path(args.out)` — 사용자가 지정한 결과 저장 경로를 Path 객체로 만듭니다.
- **L96** `output.mkdir(parents=True, exist_ok=True)` — 기록과 체크포인트를 저장할 출력 폴더를 만듭니다.
- **L97** `(output / 'metrics.json').write_text(json.dumps(report, indent=2))` — 실행 설정과 측정 기록을 JSON 파일로 저장합니다. 글의 그래프는 이 숫자에서 생성됩니다.
- **L98** `torch.save(model.state_dict(), output / 'model.pt')` — 모델의 학습된 상태를 파일에 저장합니다. 5주차에서는 어휘와 선택 step도 함께 보관합니다.
- **L99** `print('causality and batch independence: passed')` — 현재 실행의 지표나 진단 결과를 터미널에 출력합니다. JSON 로그는 후처리하기 쉽습니다.
- **L102** `if __name__ == '__main__':` — 파일을 직접 실행할 때 main을 호출하고, 다른 코드에서 import할 때는 학습을 시작하지 않도록 합니다.
- **L103** `main()` — 명령행에서 지정한 설정으로 전체 학습 실험을 실행합니다.

<a id="section-8"></a>
## 7. 학습 결과와 인과성 검증

측정 환경은 Python 3.12.14, PyTorch 2.15.0+cpu, CPU, 스레드 1개입니다. 기본 시드는 43이고 마지막 기록은 700 step입니다. 아래 숫자와 그래프는 실제 실행에서 저장한 값입니다. 시간은 기기와 병렬 작업에 따라 달라지는 참고값입니다.

| 지표 | 학습 전 | 마지막 step |
|---|---:|---:|
| 검증 교차 엔트로피 | 2.61565 | 0.00022 |
| 검증 정확도 | 12.55% | 100.00% |
| 첫 위치 평균 가중치 | 0.14828 | 0.99999 |

![검증 손실과 정확도의 학습 곡선](/attention-course/week03-learning.svg)

[원본 측정 기록](/attention-course/week03-metrics.json) · [실험별 전체 기록](/attention-course/results.json)

추가 재현 실행의 **마지막 step의 고정 검증 정확도**입니다. 데이터 시드도 함께 달라집니다.

| 시드 | 정확도 |
|---|---:|
| 43 | 100.00% |
| 53 | 100.00% |
| 63 | 100.00% |



정답 클래스가 여덟 개이므로 무작위 선택의 기준은 1/8입니다. `first_mass`는 마지막 QUERY가 첫 위치에 부여한 평균 가중치입니다. 정확도와 함께 읽되 가중치 자체를 정답 증명의 전부로 사용하지는 않습니다.

코드의 첫 검사는 마지막 두 입력을 바꾸고 그보다 앞의 logits가 같음을 확인합니다. 두 번째는 가중치 행렬의 대각선 위에 0이 아닌 값이 없는지 봅니다. 세 번째는 같은 표본을 혼자 계산할 때와 배치에 넣어 계산할 때의 출력을 비교합니다. 축을 잘못 전치해 표본끼리 정보를 섞는 버그를 발견하는 데 도움이 됩니다.

인과성이 올바르다는 것과 정답을 잘 맞힌다는 것은 독립된 조건입니다. 완전히 틀린 예측도 미래 정보 없이 만들 수 있고, 누출된 모델은 부당하게 높은 정확도를 얻을 수 있습니다. 따라서 구조적 검사와 검증 성능을 함께 확인해야 합니다.

<a id="section-9"></a>
## 8. 길이와 평가 범위

`--length 12`는 길이 12의 입력으로 **다시 학습하는** 옵션입니다. 길이 8로 학습한 모델을 그대로 12에 평가하는 길이 외삽 실험과 구분해야 합니다. 학습하지 않은 위치 임베딩은 단순히 표가 존재한다고 유효한 표현이 되는 것이 아닙니다.

길이가 늘어나면 점수 행렬의 위치 쌍이 늘어납니다. 현재처럼 모든 쌍을 명시적으로 계산하는 구현은 각 배치에서 대략 T²개의 점수를 만듭니다. 작은 실습에서는 미미하지만 실제 장문에서는 메모리와 시간이 중요해집니다.

<a id="section-10"></a>
## 9. 과제: 정보 누출을 직접 찾아내기

1. `triu(1)`을 다른 방향으로 바꾸고 미래 변경 검사가 실패하는지 확인하세요. 정확도만 보고 버그를 찾을 수 있는지도 논의합니다.
2. 위치 임베딩을 0으로 두고 모델을 다시 학습하세요. QUERY가 첫 위치를 구분할 정보가 어떤 경로에 남아 있는지 분석합니다. 인과 마스크 자체가 구조적 정보를 줄 가능성도 고려해야 합니다.
3. 손실 라벨의 무시 인덱스와 어텐션 마스크의 역할을 비교하는 예시를 작성하세요.
4. 마지막 질문의 답을 첫 기호 대신 두 번째 기호로 바꾸고 학습·평가 코드를 함께 수정하세요. 가중치 분석도 첫 위치가 아닌 정답 위치를 측정해야 합니다.

평가는 확률·타깃 시프트 설명 25점, 세 가지 구조 검사 30점, 길이·위치 실험 30점, 실패 사례와 재현성 15점입니다. 출력 행렬 그림에는 반드시 행과 열의 의미를 적습니다.

## 읽을 자료

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 특히 3.2~3.5절. 구조와 수식의 원 출처입니다.
- [PyTorch SDPA 공식 문서](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html): 마스크와 스케일링의 구현 의미를 확인합니다.
- [PyTorch CrossEntropyLoss 공식 문서](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html): logits, 정수 타깃과 무시 인덱스를 확인합니다.

이 글의 합성 데이터·실습 코드는 강의를 위해 새롭게 작성했습니다.
