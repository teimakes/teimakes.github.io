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
    """두 카드 이름과 두 속성을 질의하는 검색 문제를 생성합니다.

    카드 값은 (배치, 카드 수, 2), 질문과 정답은 각각 (배치, 2)입니다.
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
        """마지막 표현 차원을 헤드별로 나눕니다.

        (B, T, D)를 (B, H, T, D/H)로 변환합니다.
        """
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.heads, self.dh).transpose(1, 2)

    def forward(self, keys, values, query, uniform=False):
        """두 질문을 헤드별로 처리해 색과 모양의 logits를 반환합니다.

        logits: (B, 2, 4), weights: (B, H, 1, 카드 수).
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
    """기울기를 기록하지 않고 손실과 정확도를 계산합니다.

    실습별로 선택 위치, 검색 위치 또는 답 위치도 함께 측정합니다.
    """
    model.eval()
    keys, values, query, target, _ = batch
    logits, _ = model(keys, values, query, uniform)
    correct = logits.argmax(-1) == target
    return {'loss': F.cross_entropy(logits.reshape(-1, 4), target.flatten()).item(),
            'accuracy': correct.float().mean().item(),
            'joint_accuracy': correct.all(-1).float().mean().item()}


def main():
    """실행 옵션을 읽고 학습, 검증과 결과 저장을 순서대로 수행합니다."""
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
