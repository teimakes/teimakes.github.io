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
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.heads, self.dh).transpose(1, 2)

    def forward(self, keys, values, query, uniform=False):
        batch = keys.size(0)
        q_input = self.key(query).reshape(batch, 1, 2 * self.dim)
        value_input = torch.cat([self.color(values[..., 0]),
                                 self.shape(values[..., 1])], dim=-1)
        q = self.split(self.wq(q_input))
        k = self.split(self.wk(self.key(keys)))
        v = self.split(self.wv(value_input))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        weights = scores.softmax(-1)
        if uniform:
            weights = torch.ones_like(weights) / keys.size(1)
        contexts = weights @ v
        merged = contexts.transpose(1, 2).contiguous().reshape(batch, self.dim)
        logits = self.classifier(self.wo(merged)).reshape(batch, 2, 4)
        return logits, weights


@torch.no_grad()
def evaluate(model, batch, uniform=False):
    model.eval()
    keys, values, query, target, _ = batch
    logits, _ = model(keys, values, query, uniform)
    correct = logits.argmax(-1) == target
    return {'loss': F.cross_entropy(logits.reshape(-1, 4), target.flatten()).item(),
            'accuracy': correct.float().mean().item(),
            'joint_accuracy': correct.all(-1).float().mean().item()}


def main():
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
        loss = F.cross_entropy(logits.reshape(-1, 4), target.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
