---
layout: '../../layouts/Post.astro'
title: '어텐션 강의 5주차: 작은 언어 모델의 학습·평가·생성'
description: '학습 가능한 PyTorch 코드와 실습 결과를 통해 어텐션을 살펴봅니다.'
date: '2026-11-03'
tags: ['attention', 'transformer', 'pytorch', 'deep-learning']
---

> **어텐션과 트랜스포머 · 5/5**  
> 작은 트랜스포머 언어 모델을 학습하고 평가와 텍스트 생성까지 실습합니다.

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

- [이번 주의 질문: 어텐션을 실제 다음 토큰 학습으로 연결할 수 있을까?](#section-1)
- [1. 작은 언어를 설계하기](#section-2)
- [2. 입력과 정답을 한 칸씩 어긋나게 만들기](#section-3)
- [3. 디코더 블록의 순서](#section-4)
- [4. 학습 루프에서 놓치기 쉬운 점](#section-5)
- [5. 저장과 생성까지 포함한 전체 코드](#section-6)
- [6. 학습 곡선과 테스트 결과](#section-8)
- [7. 생성은 학습과 어떻게 다를까?](#section-9)
- [8. 체크포인트와 구조 검사](#section-10)
- [9. 어디까지 실제 LLM과 같고 다른가?](#section-11)
- [10. 종합 과제](#section-12)

<a id="section-1"></a>
## 이번 주의 질문: 어텐션을 실제 다음 토큰 학습으로 연결할 수 있을까?

이번 글에서는 토큰 ID부터 임베딩, 인과적 멀티헤드 셀프 어텐션, 정규화, 잔차 연결, FFN, 어휘 logits, 교차 엔트로피, 역전파, 체크포인트, 생성까지 연결합니다. 앞선 네 주의 연산이 한 개의 언어 모델 안에서 어떤 위치를 차지하는지 확인하는 종합 실습입니다.

모델은 작지만 파라미터가 실제로 업데이트됩니다. 학습 중 손실을 기록하고, 검증 손실로 최적 체크포인트를 선택하며, 별도로 분리한 테스트 조합에서 답을 생성합니다. 자연어 전체를 학습하는 대신 명세가 분명한 작은 기호 언어를 사용합니다. 따라서 필요한 정보를 복사하는 능력을 분석하기 좋고, 데이터나 대형 사전학습 모델을 별도로 내려받지 않아도 실행할 수 있습니다.

### 수업 계획과 목표

강의 90분은 언어 모델 데이터와 타깃 시프트 25분, 디코더 블록과 잔차·정규화 40분, 평가·생성·과적합 25분입니다. 실습 90분은 학습 실행, 손실 곡선 해석, 체크포인트 복원과 생성으로 구성합니다. 과제 3~4시간은 모델 절제와 일반화 분석입니다.

수업 후에는 훈련 손실만으로 모델을 평가하지 않고, 무작위인 토큰 위치의 불가피한 불확실성과 복사 정답 위치의 정확도를 구분할 수 있어야 합니다. 모델 파라미터의 저장·복원이 같은 출력을 만드는지도 확인합니다.

<a id="section-2"></a>
## 1. 작은 언어를 설계하기

한 문장은 다음 형식입니다.

```text
<BOS> A 2 B 5 ASK A ANS 2 <EOS>
<BOS> A 2 B 5 ASK B ANS 5 <EOS>
```

A와 B에는 각각 0~5의 값이 붙습니다. 질문은 A 또는 B이고 답은 해당 이름에 붙은 값입니다. 어휘는 특수 토큰과 이름·명령어·숫자를 합쳐 12개입니다. 이 실습의 토크나이저는 공백으로 구분된 이미 명세된 기호를 사전으로 정수화합니다. 실제 한국어·영어의 서브워드 토크나이저를 대체하려는 구현은 아닙니다.

숫자 조합 `(a,b)`는 36개이고 질문 종류가 두 개이므로 가능한 문장은 72개입니다. `(a+b)%5`가 0인 조합은 검증, 1인 조합은 테스트, 나머지는 학습에 사용합니다. 한 숫자 조합에 대응하는 두 질문은 같은 분할에 들어갑니다. 학습에서 본 조합의 다른 질문이 테스트에 섞이는 누출을 막기 위해서입니다.

검증은 학습 중 체크포인트 선택에 쓰고, 테스트는 선택된 모델의 최종 성능을 측정하는 데 씁니다. 이 데이터는 작고 인공적이므로 높은 테스트 정확도도 대규모 언어 이해 능력을 뜻하지 않습니다. 또한 이 결과를 보고 반복해서 설계를 조정하면 테스트 집합도 간접적으로 개발에 사용된다는 점을 기억해야 합니다.

<a id="section-3"></a>
## 2. 입력과 정답을 한 칸씩 어긋나게 만들기

각 문장의 처음 9개 토큰을 입력으로, 마지막 9개를 정답으로 사용합니다.

```text
입력: BOS A 2 B 5 ASK A ANS 2
정답: A   2 B 5 ASK A ANS 2 EOS
```

입력 위치 7은 ANS이고 이 위치의 출력이 정답 숫자를 예측합니다. 그래서 `logits[:,7]`과 원래 문장의 `data[:,8]`을 비교해 답 정확도를 구합니다. 전체 교차 엔트로피는 이 한 위치뿐 아니라 9개 위치에 적용합니다. 위치 7만 평가하는 정확도와 전체 위치의 퍼플렉서티는 서로 다른 지표입니다.

이 언어에는 본질적인 불확실성도 있습니다. A 뒤의 첫 숫자는 문장 앞부분만 보고 확정할 수 없습니다. B 뒤의 숫자와 ASK 뒤의 질문 종류도 일부 불확실합니다. 따라서 답 위치를 완벽하게 맞혀도 전체 손실이 0으로 가지 않을 수 있습니다. 가능한 조합의 분할이 만드는 상관관계도 확률 분포에 영향을 줍니다. 손실이 남는 이유를 모두 모델 실패로 해석해서는 안 됩니다.

<a id="section-4"></a>
## 3. 디코더 블록의 순서

토큰 임베딩과 위치 임베딩을 더한 표현 X가 블록으로 들어갑니다. 이 구현은 각 하위 계층 **앞**에 LayerNorm을 두는 pre-norm 형태입니다. 원래 2017년 논문의 정규화 배치와 동일한 재현 구현은 아닙니다.

```text
X₁ = X + CausalMultiHeadAttention(LayerNorm(X))
X₂ = X₁ + FFN(LayerNorm(X₁))
logits = Linear(LayerNorm(마지막 블록 출력))
```

잔차 연결은 현재 표현에 새로 계산한 변화를 더합니다. 덕분에 각 하위 계층이 전체 표현을 매번 처음부터 다시 만들 필요가 줄어듭니다. LayerNorm은 각 토큰의 특성 축을 정규화하고 학습 가능한 스케일·편향을 적용합니다. BatchNorm처럼 표본 간 통계를 공유하지 않습니다.

FFN은 위치마다 같은 가중치를 적용하는 두 선형 계층과 GELU로 구성했습니다. 첫 선형 계층은 48차원을 192차원으로 확장하고 두 번째는 다시 48로 줄입니다. 어텐션이 위치 간 정보를 섞는다면 FFN은 각 위치의 표현을 비선형으로 변환합니다. 두 역할을 구분하면 블록의 구조를 이해하기 쉽습니다.

이 예제는 차원 48, 헤드 4개, 블록 2개를 사용합니다. 드롭아웃은 넣지 않았으므로 작은 유한 데이터에서 과적합이 빠르게 나타날 수 있습니다. 그 현상 자체를 검증 곡선과 체크포인트 선택으로 관찰할 것입니다.

<a id="section-5"></a>
## 4. 학습 루프에서 놓치기 쉬운 점

`F.cross_entropy`에는 `(배치×길이, 어휘 수)` logits와 `(배치×길이,)` 정수 라벨을 전달합니다. 정답 토큰을 one-hot으로 직접 만들 필요가 없습니다. 손실 이전에 소프트맥스를 적용하지 않습니다.

`zero_grad`는 이전 기울기를 비우고 `backward`는 현재 손실의 기울기를 계산하며 `step`은 파라미터를 업데이트합니다. 그래디언트 클리핑은 큰 기울기를 제한하는 장치입니다. 학습이 항상 안정적이라는 보증은 아니므로 손실의 발산과 NaN을 함께 확인해야 합니다.

학습 표본은 학습 집합에서 배치 64개를 복원 추출합니다. 한 step은 전체 데이터를 한 번 읽는 epoch와 다릅니다. 출력 로그의 train_loss는 비교를 위해 그 시점 모델을 평가 모드로 두고 **전체 학습 문장**에서 다시 계산합니다. 검증 손실도 같은 방식으로 계산합니다. 서로 다른 배치의 우연한 손실 변화와 학습·검증 차이를 혼동하지 않도록 했습니다.

<a id="section-6"></a>
## 5. 저장과 생성까지 포함한 전체 코드

전체 소스는 [week05.py](/attention-course/week05.py)로도 받을 수 있습니다. 아래 코드는 같은 파일의 전체 내용입니다.

```bash
python week05.py
```

```python
"""Week 5: train, evaluate, save and generate with a tiny decoder LM."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

VOCAB = ['<BOS>', '<EOS>', 'A', 'B', 'ASK', 'ANS'] + [str(i) for i in range(6)]
TO_ID = {token: index for index, token in enumerate(VOCAB)}


def dataset(split='train'):
    """숫자 조합을 기준으로 학습·검증·테스트 문장을 분리합니다.

    같은 숫자 조합의 A/B 질문은 같은 분할에 배치합니다.
    """
    rows = []
    for a in range(6):
        for b in range(6):
            bucket = (a + b) % 5
            group = 'validation' if bucket == 0 else 'test' if bucket == 1 else 'train'
            if group != split:
                continue
            for query in ['A', 'B']:
                answer = a if query == 'A' else b
                tokens = ['<BOS>', 'A', str(a), 'B', str(b), 'ASK', query, 'ANS', str(answer), '<EOS>']
                rows.append([TO_ID[token] for token in tokens])
    return torch.tensor(rows, dtype=torch.long)


class Attention(nn.Module):
    def __init__(self, dim, heads, context):
        super().__init__()
        self.heads, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.projection = nn.Linear(dim, dim)
        # 마스크는 파라미터가 아닌 버퍼로 등록해 모델과 함께 저장합니다.
        self.register_buffer('future', torch.ones(context, context, dtype=torch.bool).triu(1))

    def forward(self, x):
        """인과적 멀티헤드 셀프 어텐션을 계산합니다.

        x: (B, T, D); 출력도 (B, T, D)입니다.
        """
        batch, length, dim = x.shape
        # Q·K·V를 한 번에 투영한 뒤 헤드별 차원으로 나눕니다.
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.dh)
        # 투영 결과를 Q·K·V로 분리하고 헤드 축을 앞으로 옮깁니다.
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # 각 query와 key의 내적으로 위치별 유사도를 계산합니다.
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        # 미래 위치를 소프트맥스 전에 가려 해당 가중치를 0으로 만듭니다.
        weights = scores.masked_fill(self.future[:length, :length], float('-inf')).softmax(-1)
        # 헤드별 문맥을 이어 붙여 원래 모델 차원으로 복원합니다.
        context = (weights @ v).transpose(1, 2).contiguous().reshape(batch, length, dim)
        return self.projection(context)


class Block(nn.Module):
    def __init__(self, dim, heads, context):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = Attention(dim, heads, context)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        """정규화·어텐션·FFN에 각각 잔차 연결을 적용합니다.

        입력과 출력은 모두 (B, T, D)입니다.
        """
        # 어텐션 출력에 입력을 더하는 첫 번째 잔차 연결입니다.
        x = x + self.attention(self.ln1(x))
        # FFN 출력에도 입력을 더해 두 번째 잔차 연결을 만듭니다.
        return x + self.ffn(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, dim=48, heads=4, layers=2, context=16):
        super().__init__()
        self.context = context
        self.token = nn.Embedding(len(VOCAB), dim)
        self.position = nn.Embedding(context, dim)
        self.blocks = nn.Sequential(*[Block(dim, heads, context) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, len(VOCAB))

    def forward(self, ids):
        """토큰 ID를 받아 모든 위치의 다음 토큰 logits를 반환합니다.

        ids: (B, T); 출력: (B, T, 어휘 수)입니다.
        """
        if ids.size(1) > self.context:
            raise ValueError('input exceeds context length')
        positions = torch.arange(ids.size(1), device=ids.device)
        x = self.token(ids) + self.position(positions)
        return self.head(self.norm(self.blocks(x)))

    @torch.no_grad()
    def generate(self, ids, new_tokens=2, temperature=1.0, sample=False):
        """프롬프트 뒤에 다음 토큰을 하나씩 이어 붙입니다.

        sample=False는 최대 확률 토큰을, True는 분포에서 표본을 선택합니다.
        """
        self.eval()
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        for _ in range(new_tokens):
            if ids.size(1) >= self.context:
                break
            logits = self(ids)[:, -1] / temperature
            next_id = torch.multinomial(logits.softmax(-1), 1) if sample else logits.argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        return ids


@torch.no_grad()
def evaluate(model, data):
    """기울기를 기록하지 않고 손실과 정확도를 계산합니다.

    실습별로 선택 위치, 검색 위치 또는 답 위치도 함께 측정합니다.
    """
    model.eval()
    logits = model(data[:, :-1])
    # 정답과 logits를 비교합니다. 소프트맥스는 손실 함수 내부에서 처리합니다.
    loss = F.cross_entropy(logits.reshape(-1, len(VOCAB)), data[:, 1:].reshape(-1))
    answer_accuracy = (logits[:, 7].argmax(-1) == data[:, 8]).float().mean()
    return {'loss': loss.item(), 'perplexity': loss.exp().item(),
            'accuracy': answer_accuracy.item()}


def main():
    """실행 옵션을 읽고 학습, 검증과 결과 저장을 순서대로 수행합니다."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=45)
    parser.add_argument('--out', default='runs/week05')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed + 1)
    train_data, val_data, test_data = (dataset(split) for split in ['train', 'validation', 'test'])
    # 같은 숫자 조합이 학습과 평가에 동시에 들어가지 않는지 확인합니다.
    train_pairs = {tuple(row[2:5:2].tolist()) for row in train_data}
    val_pairs = {tuple(row[2:5:2].tolist()) for row in val_data}
    assert train_pairs.isdisjoint(val_pairs)
    test_pairs = {tuple(row[2:5:2].tolist()) for row in test_data}
    assert train_pairs.isdisjoint(test_pairs) and val_pairs.isdisjoint(test_pairs)
    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    history = [{'step': 0, 'train_loss': evaluate(model, train_data)['loss'], **evaluate(model, val_data)}]
    best_loss = float('inf')
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        rows = torch.randint(len(train_data), (64,), generator=generator)
        batch = train_data[rows]
        # 입력과 정답을 한 칸씩 어긋나게 배치해 다음 토큰을 학습합니다.
        logits = model(batch[:, :-1])
        # 정답과 logits를 비교합니다. 소프트맥스는 손실 함수 내부에서 처리합니다.
        loss = F.cross_entropy(logits.reshape(-1, len(VOCAB)), batch[:, 1:].reshape(-1))
        # 이전 배치의 기울기를 비우고 현재 손실을 역전파합니다.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # 기울기가 지나치게 커질 때 업데이트의 크기를 제한합니다.
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 계산된 기울기로 학습 가능한 파라미터를 갱신합니다.
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            metrics = {'step': step, 'train_loss': evaluate(model, train_data)['loss'], **evaluate(model, val_data)}
            history.append(metrics)
            print(json.dumps(metrics))
            # 검증 손실이 가장 낮은 모델을 선택합니다. 테스트셋은 선택에 사용하지 않습니다.
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                torch.save({'model': model.state_dict(), 'vocab': VOCAB, 'step': step}, output / 'best.pt')
    checkpoint = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    prefix = test_data[:4, :8]
    generated = model.generate(prefix, new_tokens=2)
    generations = [' '.join(VOCAB[token] for token in row) for row in generated.tolist()]
    with torch.no_grad():
        reference = model(val_data[:4, :-1])
        changed = val_data[:4, :-1].clone()
        changed[:, -1] = TO_ID['0']
        assert torch.allclose(reference[:, :-1], model(changed)[:, :-1], atol=1e-5)
        reloaded = TinyLM()
        reloaded.load_state_dict(checkpoint['model'])
        reloaded.eval()
        assert torch.equal(reference, reloaded(val_data[:4, :-1]))
    report = {'torch': torch.__version__, 'seed': args.seed, 'steps': args.steps,
              'parameters': sum(p.numel() for p in model.parameters()),
              'seconds': time.perf_counter() - started, 'history': history,
              'train_sequences': len(train_data), 'val_sequences': len(val_data), 'test_sequences': len(test_data),
              'best_step': checkpoint['step'], 'best_validation': evaluate(model, val_data),
              'test': evaluate(model, test_data),
              'generations': generations}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'history'}, indent=2))


if __name__ == '__main__':
    main()
```

<a id="section-8"></a>
## 6. 학습 곡선과 테스트 결과

측정 환경은 Python 3.12.14, PyTorch 2.15.0+cpu, CPU, 스레드 1개입니다. 기본 시드는 45이고 마지막 기록은 1200 step입니다.

| 지표 | 학습 전 | 마지막 step |
|---|---:|---:|
| 검증 교차 엔트로피 | 2.71888 | 0.98709 |
| 검증 정확도 | 31.25% | 100.00% |

![검증 손실과 정확도의 학습 곡선](/attention-course/week05-learning.svg)

[원본 측정 기록](/attention-course/week05-metrics.json) · [실험별 전체 기록](/attention-course/results.json)

학습·검증·테스트 문장 수는 **42 / 16 / 14**입니다. 총 파라미터는 58,572개입니다. 검증 손실 기준으로 선택한 **100 step** 체크포인트의 검증 손실은 0.78423, 별도 테스트 손실은 0.77995, 테스트 답 정확도는 **100.00%**입니다.

테스트 프롬프트에서 두 토큰을 생성한 결과입니다. 정답 숫자는 프롬프트에 포함하지 않았습니다.

```text
<BOS> A 0 B 1 ASK A ANS 0 <EOS>
<BOS> A 0 B 1 ASK B ANS 1 <EOS>
<BOS> A 1 B 0 ASK A ANS 1 <EOS>
<BOS> A 1 B 0 ASK B ANS 0 <EOS>
```

추가 재현 실행의 **선택 체크포인트의 테스트 답 정확도**입니다. 데이터 시드도 함께 달라집니다.

| 시드 | 정확도 |
|---|---:|
| 45 | 100.00% |
| 55 | 21.43% |
| 65 | 100.00% |

![세 시드의 검증 손실과 답 정확도 비교](/attention-course/week05-seeds.svg)



마지막 step의 검증 손실보다 이전 step의 검증 손실이 더 낮을 수 있습니다. 학습 손실은 계속 줄어드는데 검증 손실이 커진다면 과적합 또는 분포 차이를 의심합니다. 코드가 `best.pt`를 저장하고 다시 읽는 이유입니다. 최적화가 끝난 마지막 파라미터와 검증으로 선택한 파라미터를 구분하세요.

위의 답 정확도는 숫자 복사 위치만 평가합니다. 퍼플렉서티는 전체 다음 토큰 분포에서 계산합니다. 답 정확도는 같아도 불확실한 위치에서 지나치게 확신하면 검증 손실이 나빠질 수 있습니다. 두 지표를 같이 기록해야 하는 이유입니다.

### 시드에 따른 일반화 성능 비교

세 실행 중 시드 55는 선택된 모델의 검증 답 정확도가 56.25%, 테스트 답 정확도가 약 21.43%였습니다. 다른 두 실행의 테스트 답 정확도 100%와 큰 차이가 납니다. 같은 구조와 학습 횟수라도 작은 조합 데이터에서 항상 원하는 검색 규칙을 배우는 것은 아닙니다. 세 결과를 비교하면 **초기화에 따라 일반화 성능이 얼마나 달라지는지** 확인할 수 있습니다.

시드 55의 학습 로그에서 학습 손실과 검증 손실을 비교해 보세요. 학습 조합의 예측을 개선하는 일이 보지 않은 조합의 복사 규칙까지 보장하지 않습니다. 답 위치의 검증 정확도와 전체 토큰의 검증 손실도 서로 다른 체크포인트를 선호할 수 있습니다. 이 차이는 손실에서 불확실한 토큰 위치가 차지하는 비중과도 관련됩니다.

분석 후 설계를 바꿀 때는 테스트 숫자를 직접 최적화하지 말고, 새로운 개발 실험에서 정한 가설을 별도 검증 집합으로 평가해야 합니다. 더 많은 값 조합, 문장 순서 변형, 데이터 증가, 여러 시드, 정규화와 목적 함수 변경을 과제로 비교할 수 있습니다. 따라서 여러 시드의 결과와 실패 사례를 함께 살펴봐야 합니다.

<a id="section-9"></a>
## 7. 생성은 학습과 어떻게 다를까?

학습 중에는 정답 문장의 앞부분을 각 위치에 입력하는 teacher forcing을 사용합니다. 생성 시에는 ANS까지의 프롬프트를 넣고 마지막 위치의 logits에서 다음 토큰을 선택합니다. 그 토큰을 입력 뒤에 붙여 다시 계산하고 EOS까지 생성합니다. 생성한 답이 틀리면 다음 계산의 입력에도 영향을 줍니다.

기본 생성은 greedy, 즉 최대 점수 토큰을 선택합니다. `sample=True`이면 확률 분포에서 샘플링하며 temperature로 분포의 날카로움을 조절합니다. 양의 temperature로 나눠도 greedy 순위는 바뀌지 않지만 샘플링 확률은 바뀝니다. 이 차이를 실제 코드에서 확인해 보세요.

이 구현은 EOS를 만났다고 행별로 조기 종료하지 않고 지정한 두 토큰만 생성합니다. 테스트 프롬프트에는 숫자와 EOS 두 토큰이 필요하므로 충분합니다. 다양한 길이의 자연어 배치 생성을 구현하려면 종료 상태 관리, 패딩·마스크, 길이 제한 등을 더 설계해야 합니다.

<a id="section-10"></a>
## 8. 체크포인트와 구조 검사

저장 파일에는 `state_dict`, 어휘 목록, 선택된 step이 들어 있습니다. 불러올 때는 같은 모델 구성으로 인스턴스를 만들고 파라미터를 복원합니다. 파일을 외부에서 받아 읽는 문제와 이 실습에서 직접 저장한 파일을 읽는 상황은 구분해야 합니다. 여기서는 `weights_only=True`를 명시합니다.

코드는 복원한 모델의 logits가 저장 모델의 출력과 정확히 같은지 확인합니다. 또 마지막 입력 숫자를 바꿨을 때 그보다 앞 위치들의 logits가 바뀌지 않는지 검사합니다. 이는 답 정확도와 별개로 구현의 인과성을 점검하는 절차입니다.

<a id="section-11"></a>
## 9. 어디까지 실제 LLM과 같고 다른가?

자기회귀 다음 토큰 학습, 토큰·위치 임베딩, 인과적 멀티헤드 어텐션, 잔차·정규화·FFN, 교차 엔트로피와 생성의 연결은 실제 디코더 언어 모델의 기본 구조와 맞닿아 있습니다. 반면 자연어 토크나이저, 대규모 데이터, 분산 학습, KV 캐시, 최적화된 어텐션 커널, 최신 위치 인코딩과 다양한 블록 변형은 범위 밖입니다.

기본 밀집 어텐션은 `(B,H,T,T)` 점수를 명시적으로 저장하므로 긴 문맥에서 비용이 커집니다. 실제 환경에서는 최적화된 SDPA 커널 등을 사용하고 메모리 접근과 수치 특성을 검토합니다. 수업의 직접 구현은 연산을 드러내기 위한 선택입니다.

<a id="section-12"></a>
## 10. 종합 과제

1. 기본 모델을 세 시드로 실행하고 최적 검증 step, 테스트 답 정확도, 전체 퍼플렉서티, 생성 예시를 보고합니다. 평균만 쓰지 말고 각 실행 결과도 남기세요.
2. 블록 수를 1과 2로 바꾸어 학습합니다. 파라미터 수와 학습 시간을 함께 비교합니다.
3. 위치 임베딩 제거 또는 FFN 제거 중 하나를 선택해 **처음부터 재학습**합니다. 학습 후 모듈을 갑자기 0으로 만드는 개입과 다른 실험임을 명시합니다.
4. 기호 언어를 세 개의 이름 A·B·C로 확장합니다. 데이터 생성, 어휘, 위치 길이, 정답 평가 위치를 모두 수정하고 분할 누출을 확인합니다.
5. 선택 과제: `F.scaled_dot_product_attention`으로 내부 연산을 바꾸고 dropout을 0으로 둔 평가 모드에서 직접 구현과 출력·기울기를 비교합니다.

평가는 전체 파이프라인 실행 30점, 구조·손실·생성 설명 25점, 독립 분할과 절제 실험 30점, 보고서·한계·재현성 15점입니다. 최종 제출물은 코드와 실행 명령, 환경 버전, 로그, 체크포인트, 결과 해석을 포함합니다.

## 읽을 자료

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 특히 3.2~3.5절. 구조와 수식의 원 출처입니다.
- [PyTorch SDPA 공식 문서](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html): 마스크와 스케일링의 구현 의미를 확인합니다.
- [PyTorch CrossEntropyLoss 공식 문서](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html): logits, 정수 타깃과 무시 인덱스를 확인합니다.

