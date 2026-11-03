---
layout: '../../layouts/Post.astro'
title: '어텐션과 트랜스포머 5: 트랜스포머 언어 모델 학습과 생성'
description: '어텐션과 트랜스포머의 계산 원리를 PyTorch 학습 실습으로 이해합니다.'
date: '2026-11-03'
tags: ['attention', 'transformer', 'pytorch', 'deep-learning']
---

> **어텐션과 트랜스포머 · 5/5**

## 목차

- [데이터 만들기](#section-1)
- [다음 토큰 정답 구성하기](#section-2)
- [디코더 블록](#section-3)
- [인과적 멀티헤드 어텐션](#section-4)
- [학습과 평가](#section-5)
- [전체 코드](#section-6)
- [실습 준비](#section-7)
- [학습과 테스트 결과](#section-8)
- [저장한 모델로 생성하기](#section-9)
- [직접 확인해 보기](#section-10)

이번 글에서는 앞에서 다룬 어텐션을 디코더 언어 모델에 넣습니다. 토큰 ID를 임베딩으로 바꾸고, 인과적 멀티헤드 셀프 어텐션과 피드포워드 계층을 통과시킨 다음, 각 위치에서 다음 토큰을 예측합니다. 학습한 모델은 검증 손실을 기준으로 저장하고 별도의 테스트 데이터로 평가합니다.

실습에는 규칙이 정해진 작은 기호 언어를 사용합니다. 모델이 학습 데이터에 없는 값의 조합에서도 앞에 나온 값을 선택할 수 있는지 확인하기 위한 데이터입니다. 자연어 문장의 품질이나 일반적인 LLM 성능을 평가하는 실험은 아닙니다.

<a id="section-1"></a>
## 데이터 만들기

각 문장에는 A와 B에 연결된 숫자가 들어갑니다. 질문은 A 또는 B 중 하나이며, 정답은 그 이름에 연결된 숫자입니다.

```text
<BOS> A 2 B 5 ASK A ANS 2 <EOS>
<BOS> A 2 B 5 ASK B ANS 5 <EOS>
```

어휘는 `<BOS>`, `<EOS>`, `A`, `B`, `ASK`, `ANS`와 숫자 0~5를 합쳐 12개입니다. 실습에서는 이 기호들을 사전으로 정수 ID에 대응시킵니다. 일반적인 자연어 토크나이저는 사용하지 않습니다.

숫자 쌍 `(a, b)`는 36가지이고 질문은 A 또는 B이므로 가능한 문장은 72개입니다. 코드에서는 `(a+b) % 5`를 기준으로 숫자 쌍을 학습·검증·테스트에 나눕니다. 같은 숫자 쌍에 대한 A 질문과 B 질문은 같은 분할에 넣습니다. 모델이 학습에서 본 쌍의 다른 질문만 테스트에 등장하는 상황을 피하기 위해서입니다.

학습 중에는 검증 손실을 확인해 저장할 체크포인트를 고릅니다. 테스트 집합은 선택된 모델을 평가할 때 사용합니다. 다만 테스트 문장 수가 적고 반복 실험 과정에서 결과를 살펴봤으므로, 그 수치를 대규모 데이터에 대한 일반화 성능으로 해석할 수는 없습니다.

<a id="section-2"></a>
## 다음 토큰 정답 구성하기

한 문장을 다음처럼 한 위치씩 어긋나게 나누면 각 입력 위치의 다음 토큰이 정답이 됩니다.

```text
입력: BOS A 2 B 5 ASK A ANS 2
정답: A   2 B 5 ASK A ANS 2 EOS
```

모델은 모든 입력 위치에서 어휘 전체에 대한 logits를 출력합니다. 교차 엔트로피는 각 위치의 logits와 정답 ID를 비교해 계산합니다. `ANS` 위치의 출력은 질문에 해당하는 숫자를 예측하는 데 사용됩니다.

모든 위치의 손실과 답 위치의 정확도는 다른 지표입니다. 예를 들어 첫 번째 숫자가 등장하기 전에는 그 숫자를 앞 문맥만으로 확정할 수 없습니다. 따라서 답 위치의 정확도가 높더라도 전체 다음 토큰 손실이 0이 될 필요는 없습니다.

<a id="section-3"></a>
## 디코더 블록

이 모델은 토큰 임베딩과 학습 가능한 위치 임베딩을 더한 뒤 두 개의 디코더 블록을 통과시킵니다. 각 블록은 다음 순서로 계산합니다.

```text
h = x + causal_attention(layer_norm(x))
output = h + feed_forward(layer_norm(h))
```

어텐션은 위치 간 정보를 결합하고 피드포워드 신경망은 각 위치의 표현을 변환합니다. 두 계층에는 각각 잔차 연결이 있습니다. LayerNorm을 각 계층 앞에 두는 **pre-norm** 구조이며, 원래 트랜스포머 논문의 post-norm 구조를 그대로 재현한 것은 아닙니다.

모델 차원은 48, 헤드 수는 4이므로 헤드당 차원은 12입니다. 피드포워드 계층은 48차원을 192차원으로 확장하고 GELU를 적용한 다음 다시 48차원으로 변환합니다. 최종 선형 계층은 각 위치의 표현을 12개 어휘에 대한 logits로 바꿉니다.

<a id="section-4"></a>
## 인과적 멀티헤드 어텐션

`Attention`은 하나의 선형 계층으로 Q·K·V를 계산합니다. 그 결과를 `(배치, 길이, 3, 헤드 수, 헤드 차원)`으로 재배치하고 Q·K·V 축을 분리합니다. 각 텐서의 모양은 `(배치, 헤드 수, 길이, 헤드 차원)`이 됩니다.

Query와 Key의 내적을 헤드 차원의 제곱근으로 나누고, 미래 위치의 점수를 음의 무한대로 바꿉니다. 그다음 위치별 소프트맥스를 적용하고 Value의 가중합을 계산합니다. 헤드 출력을 다시 결합하면 입력과 같은 모델 차원의 표현을 얻습니다.

```text
scores  = QKᵀ / √dₕ
weights = softmax(mask(scores))
context = weights V
```

마스크는 `register_buffer()`로 모델에 등록합니다. 학습으로 갱신되는 파라미터는 아니지만 모델의 상태와 함께 보관됩니다. 현재 구현은 점수 행렬을 직접 만들기 때문에 계산 과정을 확인하기 쉽습니다. 긴 문맥에서 사용할 때는 `(길이 × 길이)` 점수 행렬의 계산과 저장 비용을 고려해야 합니다.

<a id="section-5"></a>
## 학습과 평가

학습 루프는 다음 순서로 실행됩니다.

1. 학습 문장에서 배치를 추출하고 입력과 정답을 한 위치씩 나눕니다.
2. 모델이 출력한 logits와 정답 ID로 교차 엔트로피를 계산합니다.
3. 이전 기울기를 비운 뒤 `backward()`로 기울기를 계산합니다.
4. 기울기 크기를 제한하고 AdamW로 파라미터를 갱신합니다.
5. 일정 단계마다 전체 학습 집합과 검증 집합의 지표를 계산합니다.

로그의 `train_loss`는 해당 시점의 전체 학습 문장에 대한 손실입니다. `loss`와 `perplexity`는 검증 집합의 전체 다음 토큰 예측을 기준으로 합니다. `accuracy`는 질문에 해당하는 숫자를 맞혔는지 평가합니다.

<a id="section-6"></a>
## 전체 코드

<a id="section-7"></a>
## 실습 준비

Python 3.12에서 실행할 수 있습니다. PyTorch를 설치한 뒤 [week05.py](/attention-course/week05.py)를 내려받아 실행하세요. 아래 코드와 내려받는 파일은 같습니다.

```bash
python -m pip install torch==2.15.0
python week05.py
```

결과는 기본적으로 `runs/week05`에 저장됩니다. `--seed`, `--steps`, `--out` 옵션으로 시드, 학습 단계, 저장 경로를 지정할 수 있습니다.

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
## 학습과 테스트 결과

기본 설정은 시드 45, 1200단계 학습입니다.

| 검증 지표 | 학습 전 | 마지막 단계 |
|---|---:|---:|
| 교차 엔트로피 | 2.71888 | 0.98709 |
| 정확도 | 31.25% | 100.00% |

![검증 손실과 정확도 그래프](/attention-course/week05-learning.svg)

학습·검증·테스트 문장은 각각 42, 16, 14개입니다. 검증 손실로 선택한 단계는 100이며 테스트 답 정확도는 100.00%입니다.

테스트 입력에서 생성한 결과는 다음과 같습니다.

```text
<BOS> A 0 B 1 ASK A ANS 0 <EOS>
<BOS> A 0 B 1 ASK B ANS 1 <EOS>
<BOS> A 1 B 0 ASK A ANS 1 <EOS>
<BOS> A 1 B 0 ASK B ANS 0 <EOS>
```

| 시드 | 선택 모델의 테스트 답 정확도 |
|---|---:|
| 45 | 100.00% |
| 55 | 21.43% |
| 65 | 100.00% |

![세 시드의 검증 손실과 답 정확도](/attention-course/week05-seeds.svg)

[측정 기록 보기](/attention-course/week05-metrics.json)


학습 손실이 계속 감소하더라도 검증 손실은 다시 증가할 수 있습니다. 이 코드가 검증 손실이 가장 낮았을 때의 모델을 저장하는 이유입니다. 마지막 단계의 파라미터와 검증 기준으로 선택한 파라미터는 다를 수 있습니다.

세 시드 중 55에서는 선택된 모델의 검증 답 정확도가 56.25%, 테스트 답 정확도가 약 21.43%였습니다. 나머지 두 시드에서는 테스트 답을 모두 맞혔습니다. 작은 데이터에서는 초기값에 따라 학습 조합을 처리하는 방식과 새로운 조합에 대한 성능이 달라질 수 있으므로, 한 번의 결과만으로 모델의 안정성을 판단하기 어렵습니다.

<a id="section-9"></a>
## 저장한 모델로 생성하기

`best.pt`에는 모델의 상태, 어휘, 선택된 학습 단계가 들어 있습니다. 코드는 같은 구조의 모델을 다시 만들고 상태를 불러온 뒤, 복원 전후의 logits가 같은지 검사합니다. 이 실습은 직접 저장한 체크포인트를 `weights_only=True`로 읽습니다.

생성할 때는 `ANS`까지를 입력합니다. 모델이 출력한 마지막 위치의 logits에서 다음 토큰을 선택하고 입력 뒤에 붙인 다음 같은 과정을 반복합니다. 기본 설정은 가장 높은 점수의 토큰을 고르는 greedy 방식입니다. `sample=True`를 지정하면 확률에 따라 표본을 선택하며, 이때 temperature로 분포를 조절할 수 있습니다.

코드는 지정한 수만큼 토큰을 생성합니다. 각 표본이 EOS에 도달했는지 따로 확인해 중단하는 기능과 KV 캐시는 구현하지 않았습니다.

<a id="section-10"></a>
## 직접 확인해 보기

1. 세 시드의 학습 손실, 검증 손실, 테스트 답 정확도를 비교합니다. 검증 손실이 가장 낮은 단계와 답 정확도가 가장 높은 단계가 같은지 확인합니다.
2. 블록 수를 1과 2로 바꾸어 각각 처음부터 학습합니다. 파라미터 수와 테스트 성능을 함께 기록합니다.
3. A와 B 외에 C를 추가합니다. 데이터 생성, 어휘, 입력 길이, 답의 평가 위치, 데이터 분할 검사를 함께 수정합니다.
4. 직접 구현한 어텐션을 PyTorch의 `scaled_dot_product_attention()`으로 교체하고 출력과 기울기를 비교합니다. 마스크의 의미와 평가 시 dropout 설정을 먼저 확인합니다.

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
