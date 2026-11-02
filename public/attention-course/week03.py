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
