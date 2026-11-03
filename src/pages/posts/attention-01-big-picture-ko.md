---
layout: '../../layouts/Post.astro'
title: '어텐션 강의 1주차: 선택 가중치를 학습하는 원리'
description: '학습 가능한 PyTorch 코드와 실습 결과를 통해 어텐션을 살펴봅니다.'
date: '2026-11-03'
tags: ['attention', 'transformer', 'pytorch', 'deep-learning']
---

> **어텐션과 트랜스포머 · 1/5**  
> 분류 과제를 통해 어텐션이 중요한 입력을 선택하는 원리를 살펴봅니다.

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

- [이번 주의 질문: 무엇을 볼지도 학습할 수 있을까?](#section-1)
- [1. LLM까지의 큰 그림](#section-2)
- [2. 통제된 문제를 만드는 이유](#section-3)
- [3. 점수에서 확률로, 확률에서 문맥으로](#section-4)
- [4. 어텐션 점수까지 기울기가 도달하는 이유](#section-5)
- [5. 전체 구현과 텐서 모양](#section-6)
- [6. 학습 결과 읽기](#section-8)
- [7. 실험과 디버깅](#section-9)
- [8. 과제와 평가 기준](#section-10)

<a id="section-1"></a>
## 이번 주의 질문: 무엇을 볼지도 학습할 수 있을까?

긴 문서를 읽고 질문에 답할 때 문서의 모든 단어가 같은 중요도를 가지지는 않습니다. 영화 리뷰의 긍정·부정을 분류할 때도 영화 제목과 감정 표현의 역할은 다릅니다. 여러 벡터에서 필요한 정보를 모으는 가장 단순한 방법은 평균이지만, 평균은 모든 위치를 같은 비중으로 취급합니다. 어텐션은 이 비중을 입력과 목적에 따라 계산하도록 만듭니다.

이번 주에는 트랜스포머를 바로 구현하지 않습니다. 먼저 **무엇을 볼지 결정하는 계산도 정답을 맞히는 과정에서 학습될 수 있다**는 사실을 실험합니다. 데이터 생성, 정답 정의, 점수 함수, 가중합, 분류 손실, 역전파가 어떻게 연결되는지 이해하면 다음 주의 Q·K·V가 갑자기 등장한 기호처럼 보이지 않습니다.

### 수업 구성과 도달 목표

한 주를 강의 90분, 실습 90분, 과제 2~3시간으로 구성했습니다. 선형대수의 벡터·행렬곱과 Python 함수·클래스를 알고 있다고 가정합니다. 자동 미분은 이 글에서 필요한 범위부터 설명합니다.

- 강의 전반 45분: 평균 풀링과 학습 가능한 가중합을 비교하고 데이터 모양을 추적합니다.
- 강의 후반 45분: 소프트맥스 미분과 정답 오차가 점수 함수까지 전달되는 경로를 유도합니다.
- 실습 전반 45분: 전체 코드를 실행하고 학습 전후의 성능과 선택 가중치를 비교합니다.
- 실습 후반 45분: 균등 가중치 개입과 시드·길이 변경 실험을 수행합니다.
- 수업 후: 어텐션 지도 학습 여부와 검증 데이터의 독립성을 설명하는 짧은 실험 보고서를 제출합니다.

수업을 마치면 `(B,T,D)`에서 각 축이 무엇인지 설명하고, 가중치가 합이 1인 이유와 분류 손실만으로 점수 함수가 바뀌는 이유를 직접 보여 줄 수 있어야 합니다.

<a id="section-2"></a>
## 1. LLM까지의 큰 그림

텍스트는 토크나이저를 거쳐 정수 ID가 되고, 임베딩이 각 ID를 벡터로 바꿉니다. 트랜스포머 블록은 여러 위치의 정보를 섞는 어텐션과 각 위치의 표현을 변환하는 피드포워드 신경망을 반복합니다. 마지막 출력 계층은 다음 토큰 후보의 점수를 계산합니다. 학습은 정답 토큰에 높은 확률을 주도록 이 경로의 파라미터를 함께 조정하는 과정입니다.

여기서 이번 주가 다루는 범위는 **여러 입력 벡터를 하나의 문맥 벡터로 모으는 단계**입니다. 입력을 섞는 규칙 자체가 미분 가능하므로 뒤에 있는 분류기의 오차가 앞의 중요도 계산까지 전달됩니다. 언어 의미를 해석하기 전에 이 연결부터 확인할 것입니다.

![토큰 입력부터 다음 토큰 점수까지, 잔차 연결을 포함한 디코더의 큰 그림](/attention-course/architecture.svg)

그림의 주황색 상자 안에서 ‘어디에서 얼마나 읽을지’가 계산됩니다. 첫 주에는 그 상자의 핵심인 선택과 가중합을 따로 학습하고, 5주차에서 그림 전체를 실제 코드로 연결합니다. 그림은 이 강의의 pre-norm 디코더를 나타냅니다.

<a id="section-3"></a>
## 2. 통제된 문제를 만드는 이유

입력이 여덟 개의 작은 레코드로 이루어져 있다고 합시다. 각 레코드는 `[표식, 수치]`라는 두 특성을 가집니다. 표식이 1인 레코드는 정확히 하나이고 나머지는 0입니다. 정답은 **표식이 있는 레코드의 수치가 양수인지**입니다. 나머지 수치는 정답과 무관한 방해 정보입니다.

예를 들어 `[0,-0.3], [1,+0.8], [0,-1.4], …`의 정답은 긍정 클래스 1입니다. 중요한 위치는 매번 무작위로 바뀝니다. 항상 두 번째 레코드만 보거나 모든 수치를 더하는 것으로는 안정적으로 풀 수 없습니다. 중요한 것은 특정 단어를 외우는 능력이 아니라 **표식에 해당하는 정보를 찾아 읽는 규칙**입니다.

이 문제에는 학습용 위치 정답이 존재합니다. 그러나 코드에서는 선택 위치를 손실 함수에 넣지 않습니다. 선택 위치는 나중에 분석 지표를 계산할 때만 사용합니다. 모델이 받는 감독 신호는 최종 이진 분류 정답 하나뿐입니다. 따라서 선택 가중치가 표식 위치에 몰린다면 그것은 분류 오차를 줄이는 과정에서 생긴 현상입니다.

검증 데이터는 학습과 다른 난수 생성기로 한 번 생성해 고정합니다. 매 학습 단계에는 새로운 학습 표본을 생성합니다. 이는 유한한 문장 목록을 외우는 효과를 줄이지만, 같은 생성 규칙에서 뽑힌 표본이라는 범위 안의 일반화입니다. 실제 문서 이해나 새로운 과제에 대한 일반화까지 입증하지는 않습니다.

<a id="section-4"></a>
## 3. 점수에서 확률로, 확률에서 문맥으로

위치 i의 입력을 xᵢ, 학습 가능한 점수 함수를 sθ라고 합시다. 작은 신경망이 각 위치에 실수 점수를 매깁니다. 그 점수는 확률이 아니므로 음수가 나와도 괜찮습니다.

```text
sᵢ = w₂ᵀ tanh(W₁xᵢ + b₁) + b₂
αᵢ = exp(sᵢ) / Σⱼ exp(sⱼ)
vᵢ = Wᵥxᵢ + bᵥ
c  = Σᵢ αᵢ vᵢ
z  = Wₒc + bₒ
L  = −log softmax(z)[정답 클래스]
```

소프트맥스의 분모는 같은 표본 안의 모든 위치를 합한 것입니다. 표본끼리 합하면 다른 사람의 입력이 내 가중치에 영향을 주는 오류가 생깁니다. `dim=-1`은 점수 텐서 `(B,T)`의 위치 축 T를 정규화합니다.

소프트맥스는 점수의 절대값보다 상대 차이에 반응합니다. 모든 점수에 같은 상수를 더해도 비율은 같습니다. 하나의 점수만 크게 올리면 그 위치의 가중치가 커지고 나머지 위치의 가중치는 함께 줄어듭니다. `softmax`를 한 뒤에는 음수가 없고 위치별 합은 1입니다.

가중합 c는 vᵢ들과 같은 차원의 벡터입니다. 가중치는 어디에서 가져올지 결정하고, value 변환은 무엇을 가져올지 결정합니다. 둘을 분리했기 때문에 같은 중요도 규칙을 쓰더라도 전달 정보의 표현을 학습할 수 있습니다.

<a id="section-5"></a>
## 4. 어텐션 점수까지 기울기가 도달하는 이유

소프트맥스 미분은 `∂αᵢ/∂sⱼ = αᵢ(δᵢⱼ−αⱼ)`입니다. δ는 i와 j가 같을 때 1, 다르면 0입니다. 하나의 점수를 바꾸면 자기 가중치뿐 아니라 모든 위치의 비율이 달라진다는 사실이 식에도 나타납니다.

문맥 벡터 c에 대해 손실의 기울기를 g라고 놓으면 다음 관계를 얻습니다.

```text
g = ∂L/∂c
∂L/∂sᵢ = αᵢ · gᵀ(vᵢ − c)
```

이 식은 중요합니다. 위치 i의 정보가 현재 평균 문맥 c와 얼마나 다르고, 그 차이가 손실을 줄이는 방향인지에 따라 점수가 조정됩니다. 모델은 ‘여기를 봐라’라는 위치 정답을 직접 받지 않아도, 어떤 정보를 더 가져오면 분류가 좋아지는지 기울기를 통해 조정할 수 있습니다.

코드에서 `loss.backward()`가 이 연쇄 법칙을 자동으로 적용합니다. `optimizer.step()`은 계산된 기울기로 파라미터를 바꿉니다. `backward()`만 호출하고 `step()`을 빼면 기울기는 계산되지만 모델은 학습되지 않습니다. 반대로 이전 단계의 기울기를 비우지 않으면 의도하지 않은 누적이 발생합니다.

<a id="section-6"></a>
## 5. 전체 구현과 텐서 모양

- `inputs`: `(128,8,2)` — 배치, 위치, 입력 특성
- `scores`, `weights`: `(128,8)`
- `self.value(inputs)`: `(128,8,16)`
- `context`: `(128,16)`
- `logits`: `(128,2)` / `labels`: `(128,)`, 정수 클래스 ID

분류 손실에는 소프트맥스 적용 전의 logits를 전달합니다. PyTorch의 교차 엔트로피가 내부적으로 로그 소프트맥스를 처리하므로, 확률을 한 번 더 넣으면 의도한 목적 함수가 달라집니다.

전체 소스는 [week01.py](/attention-course/week01.py)로도 받을 수 있습니다. 아래 코드는 같은 파일의 전체 내용입니다.

```bash
python week01.py
```

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

<a id="section-8"></a>
## 6. 학습 결과 읽기

측정 환경은 Python 3.12.14, PyTorch 2.15.0+cpu, CPU, 스레드 1개입니다. 기본 시드는 41이고 마지막 기록은 700 step입니다.

| 지표 | 학습 전 | 마지막 step |
|---|---:|---:|
| 검증 교차 엔트로피 | 0.68371 | 0.02466 |
| 검증 정확도 | 56.54% | 99.85% |
| 정답 위치 평균 가중치 | 0.15354 | 0.98967 |

![검증 손실과 정확도의 학습 곡선](/attention-course/week01-learning.svg)

[원본 측정 기록](/attention-course/week01-metrics.json) · [실험별 전체 기록](/attention-course/results.json)

학습 후 균등 가중치로 개입했을 때 정확도는 **60.94%**였습니다. 이는 동일 구조의 균등 모델을 처음부터 학습한 결과와는 구분해야 합니다.

추가 재현 실행의 **마지막 step의 고정 검증 정확도**입니다. 데이터 시드도 함께 달라집니다.

| 시드 | 정확도 |
|---|---:|
| 41 | 99.85% |
| 51 | 99.66% |
| 61 | 99.76% |



`accuracy`는 이진 분류 정확도이고 `selected_mass`는 표식이 있는 위치에 할당한 평균 가중치입니다. 학습 전에는 분류 정확도가 대체로 우연 수준에 가깝고, 위치별 가중치도 넓게 퍼질 수 있습니다. 학습이 진행되면 분류 손실과 검증 정확도, 중요한 위치에 주는 가중치가 어떤 관계로 변하는지 그래프를 함께 봅니다.

균등 가중치 개입은 **이미 학습한 모델에서 가중치만 강제로 같게 바꾼 실험**입니다. 균등 풀링 모델을 처음부터 따로 학습한 공정한 기준선 비교와는 다릅니다. 성능이 떨어지면 이 모델이 학습한 선택 규칙을 사용하고 있었다는 근거가 되지만, 어텐션이 모든 문제에서 다른 구조보다 우월하다는 결론은 낼 수 없습니다.

분류 정확도가 높아도 표식 위치의 가중치가 반드시 1일 필요는 없습니다. 필요한 신호가 충분히 전달되면 뒤의 분류기가 해결할 수 있기 때문입니다. ‘정답 위치의 가중치가 가장 높다’와 ‘모델의 모든 판단 이유를 설명했다’도 구분해야 합니다.

<a id="section-9"></a>
## 7. 실험과 디버깅

1. `make_batch`의 기본 길이를 8에서 16으로 바꿔 다시 학습하세요. 나머지 조건을 고정하고 최종 정확도와 선택 가중치의 변화를 비교합니다.
2. 표식 특성을 모든 위치에서 0으로 만들면 어떤 일이 생길지 먼저 예측하세요. 정답 정의는 원래 선택 위치를 유지합니다. 관측 입력에서 중요한 위치를 알 수 없을 때 모델이 할 수 있는 일이 무엇인지 설명합니다.
3. `scorer gradient norm`이 0인지 확인합니다. `detach()`를 가중치에 삽입해 기울기 경로를 끊고 차이를 관찰하세요.
4. 균등 풀링 모델을 **처음부터** 학습하는 비교 실험을 별도로 작성합니다. 학습 후 개입 실험과 다른 질문에 답한다는 점을 보고서에 명시하세요.

손실이 줄지 않으면 먼저 `labels`의 dtype과 값 범위, logits의 모양을 확인합니다. `model.eval()`은 평가 모드를 설정하고 `torch.no_grad()`는 그래디언트 기록을 비활성화합니다. 두 동작은 목적이 다르며, 학습 루프에서는 반드시 다시 `model.train()`으로 돌아갑니다.

<a id="section-10"></a>
## 8. 과제와 평가 기준

제출물은 실행 명령, 세 개의 시드 결과, 학습 곡선, 잘된 예와 실패 예 각 세 개, 해석을 담은 2~3쪽 보고서입니다. ‘잘된다’는 문장 대신 정확도와 오차를 수치로 제시하세요.

- 이론 30점: 소프트맥스 미분과 점수 기울기의 의미를 자기 말로 설명합니다.
- 구현 30점: 전체 코드가 새 환경에서 실행되고 텐서 축과 학습·평가 모드가 올바릅니다.
- 실험 25점: 시드·입력 길이·균등 풀링 비교에서 조건을 통제합니다.
- 해석 15점: 통제된 합성 과제의 성능을 언어 이해 성능으로 확대하지 않습니다.

다음 주에는 점수 함수에 명시적인 query를 추가해, 질문에 따라 서로 다른 레코드를 찾도록 확장합니다.

## 읽을 자료

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 특히 3.2~3.5절. 구조와 수식의 원 출처입니다.
- [PyTorch SDPA 공식 문서](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html): 마스크와 스케일링의 구현 의미를 확인합니다.
- [PyTorch CrossEntropyLoss 공식 문서](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html): logits, 정수 타깃과 무시 인덱스를 확인합니다.

