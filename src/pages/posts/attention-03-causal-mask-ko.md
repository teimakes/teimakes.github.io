---
layout: '../../layouts/Post.astro'
title: '어텐션 강의 3주차: 인과적 셀프 어텐션과 정보 누출'
description: '학습 가능한 PyTorch 코드와 실습 결과를 통해 어텐션을 살펴봅니다.'
date: '2026-11-03'
tags: ['attention', 'transformer', 'pytorch', 'deep-learning']
---

> **어텐션과 트랜스포머 · 3/5**  
> 미래 정보를 가리는 원리와 인과적 셀프 어텐션의 학습을 살펴봅니다.

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

- [이번 주의 질문: 병렬로 학습하면서도 미래를 못 보게 할 수 있을까?](#section-1)
- [1. 자기회귀 언어 모델의 정보 범위](#section-2)
- [2. 지연 복사 과제 명세](#section-3)
- [3. 셀프 어텐션과 위치 임베딩](#section-4)
- [4. 마스크의 행과 열을 확실히 구분하기](#section-5)
- [5. 전체 PyTorch 코드](#section-6)
- [6. 학습 결과와 인과성 검증](#section-8)
- [7. 길이와 평가 범위](#section-9)
- [8. 과제: 정보 누출을 직접 찾아내기](#section-10)

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
    """임의 기호와 마지막 QUERY로 입력을 만들고 첫 기호를 복사할 정답으로 둡니다.

    입력과 라벨은 (배치, 길이)이며 QUERY 외의 라벨은 -100으로 무시합니다.
    """
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
        """미래 위치를 가리는 셀프 어텐션으로 각 위치의 logits를 계산합니다.

        ids: (B, T), logits: (B, T, 8), weights: (B, T, T).
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
    """기울기를 기록하지 않고 손실과 정확도를 계산합니다.

    실습별로 선택 위치, 검색 위치 또는 답 위치도 함께 측정합니다.
    """
    model.eval()
    x, y = batch
    logits, weights = model(x)
    return {'loss': F.cross_entropy(logits.flatten(0, 1), y.flatten()).item(),
            'accuracy': (logits[:, -1].argmax(-1) == y[:, -1]).float().mean().item(),
            'first_mass': weights[:, -1, 0].mean().item()}


def main():
    """실행 옵션을 읽고 학습, 검증과 결과 저장을 순서대로 수행합니다."""
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

<a id="section-8"></a>
## 6. 학습 결과와 인과성 검증

측정 환경은 Python 3.12.14, PyTorch 2.15.0+cpu, CPU, 스레드 1개입니다. 기본 시드는 43이고 마지막 기록은 700 step입니다.

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
## 7. 길이와 평가 범위

`--length 12`는 길이 12의 입력으로 **다시 학습하는** 옵션입니다. 길이 8로 학습한 모델을 그대로 12에 평가하는 길이 외삽 실험과 구분해야 합니다. 학습하지 않은 위치 임베딩은 단순히 표가 존재한다고 유효한 표현이 되는 것이 아닙니다.

길이가 늘어나면 점수 행렬의 위치 쌍이 늘어납니다. 현재처럼 모든 쌍을 명시적으로 계산하는 구현은 각 배치에서 대략 T²개의 점수를 만듭니다. 작은 실습에서는 미미하지만 실제 장문에서는 메모리와 시간이 중요해집니다.

<a id="section-10"></a>
## 8. 과제: 정보 누출을 직접 찾아내기

1. `triu(1)`을 다른 방향으로 바꾸고 미래 변경 검사가 실패하는지 확인하세요. 정확도만 보고 버그를 찾을 수 있는지도 논의합니다.
2. 위치 임베딩을 0으로 두고 모델을 다시 학습하세요. QUERY가 첫 위치를 구분할 정보가 어떤 경로에 남아 있는지 분석합니다. 인과 마스크 자체가 구조적 정보를 줄 가능성도 고려해야 합니다.
3. 손실 라벨의 무시 인덱스와 어텐션 마스크의 역할을 비교하는 예시를 작성하세요.
4. 마지막 질문의 답을 첫 기호 대신 두 번째 기호로 바꾸고 학습·평가 코드를 함께 수정하세요. 가중치 분석도 첫 위치가 아닌 정답 위치를 측정해야 합니다.

평가는 확률·타깃 시프트 설명 25점, 세 가지 구조 검사 30점, 길이·위치 실험 30점, 실패 사례와 재현성 15점입니다. 출력 행렬 그림에는 반드시 행과 열의 의미를 적습니다.

## 읽을 자료

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 특히 3.2~3.5절. 구조와 수식의 원 출처입니다.
- [PyTorch SDPA 공식 문서](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html): 마스크와 스케일링의 구현 의미를 확인합니다.
- [PyTorch CrossEntropyLoss 공식 문서](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html): logits, 정수 타깃과 무시 인덱스를 확인합니다.

