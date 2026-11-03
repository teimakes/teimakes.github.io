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
    """한 이름의 색과 다른 이름의 모양을 동시에 질문합니다.

    keys: (B, 5). 표본마다 중복되지 않는 이름 다섯 개입니다.
    values: (B, 5, 2). 마지막 축은 색, 모양이며 각각 0~3입니다.
    chosen: (B, 2). 첫 질문과 두 번째 질문의 항목 위치입니다.
    query = keys.gather(1, chosen): (B, 2) 질문 이름을 구합니다.
    target의 첫 열은 첫 질문 항목의 색, 둘째 열은 두 번째 항목의 모양입니다.
    stack(..., dim=-1)으로 두 정답을 (B, 2)로 결합합니다.

    두 질문이 같은 항목을 가리킬 수도 있습니다. chosen은 학습 손실이
    아니라 가중치 분석에 사용하도록 정답과 함께 반환합니다.
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
        """같은 총 차원에서 헤드 수를 바꿀 수 있는 검색 모델을 만듭니다.

        dim은 heads로 나누어져야 하며 dh=dim/heads입니다.
        key는 이름 12종, color와 shape은 각 속성 4종의 임베딩입니다.
        wq는 두 질문 임베딩을 합친 2*dim차원을 dim으로 변환합니다.
        wk는 이름 임베딩, wv는 두 속성의 결합을 각각 dim으로 변환합니다.
        wo는 결합된 헤드 출력을 투영합니다.
        classifier는 색 4개와 모양 4개의 점수를 합쳐 8개 출력합니다.

        heads만 바꾸면 각 헤드의 차원은 달라지지만 전체 파라미터 수는 같습니다.
        """
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
        """전체 표현을 헤드별 표현으로 나누고 축 순서를 바꿉니다.

        x: (B, T, D).
        reshape(B, T, H, Dh): 마지막 D차원을 H*Dh로 나눕니다.
        transpose(1, 2): 결과를 (B, H, T, Dh) 순서로 바꿉니다.
        이 배치에서는 헤드별 행렬곱을 서로 독립적으로 계산할 수 있습니다.
        """
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.heads, self.dh).transpose(1, 2)

    def forward(self, keys, values, query, uniform=False):
        """두 질문에서 만든 Query로 항목의 속성을 검색합니다.

        keys: (B, 5), values: (B, 5, 2), query: (B, 2)입니다.
        q_input: 두 이름 임베딩을 순서대로 이어 (B, 1, 2D)로 만듭니다.
            질문이 두 개여도 이 구현의 Query 길이는 1입니다.
        value_input: 색·모양 임베딩을 합쳐 (B, 5, 2D)로 만듭니다.
        q/k/v: 투영과 split 후 (B,H,1,Dh), (B,H,5,Dh), (B,H,5,Dh)입니다.
        scores: Q·K 내적을 sqrt(Dh)로 나눈 (B,H,1,5) 텐서입니다.
        weights: 다섯 항목 축을 정규화합니다. 헤드 축은 정규화하지 않습니다.
        contexts: Value 가중합이며 (B,H,1,Dh)입니다.
        merged: 축을 되돌려 헤드들을 이어 붙인 (B,D) 문맥입니다.
        wo와 classifier: 이를 색과 모양의 (B,2,4) logits로 바꿉니다.

        반환한 weights에서 각 헤드가 두 질문 중 어느 항목에 가중치를
        주는지 확인할 수 있습니다. 헤드 역할은 미리 지정하지 않습니다.
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
    """항목별 정확도와 두 답을 모두 맞힌 비율을 측정합니다.

    logits.reshape(-1, 4)는 (B,2,4)를 (B*2,4)로 변환합니다.
    target.flatten()은 같은 순서의 (B*2,) 라벨입니다.
    cross_entropy는 두 답에 대한 평균 손실입니다.
    correct는 (B,2)이며 각 답의 정답 여부를 표시합니다.
    accuracy는 모든 답의 평균 정확도입니다.
    joint_accuracy는 correct.all(-1), 즉 두 답이 모두 맞은 표본의 비율입니다.

    uniform을 지정하면 모든 헤드에서 항목 가중치를 동일하게 바꾸어 평가합니다.
    """
    model.eval()
    keys, values, query, target, _ = batch
    logits, _ = model(keys, values, query, uniform)
    correct = logits.argmax(-1) == target
    return {'loss': F.cross_entropy(logits.reshape(-1, 4), target.flatten()).item(),
            'accuracy': correct.float().mean().item(),
            'joint_accuracy': correct.all(-1).float().mean().item()}


def main():
    """헤드 수에 따른 두 속성 검색 모델의 학습 결과를 기록합니다.

    --heads는 총 차원 32를 나눌 헤드 수입니다. 다른 조건을 유지한
    헤드 수 비교가 가능하도록 임베딩과 전체 투영 크기는 바꾸지 않습니다.
    검증 2,048개를 고정하고 매 단계 학습 표본 128개를 새로 생성합니다.
    두 답의 logits를 (B*2,4)로, target을 (B*2,)로 바꿔 손실을 계산합니다.
    역전파와 AdamW 갱신을 수행하고 200단계마다 항목별·공동 정확도를 기록합니다.

    마지막에는 가중치의 (1,H,1,5) 모양과 항목 축 합을 검사합니다.
    파라미터 수, 헤드별 표본 가중치, 균등 개입 결과를 metrics.json에
    함께 저장하고 model.pt에 모델 상태를 보관합니다.
    """
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
