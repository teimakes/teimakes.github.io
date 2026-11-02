"""Week 1: learn attention pooling from classification supervision."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def make_batch(size, generator, length=8):
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
        scores = self.scorer(inputs).squeeze(-1)
        weights = scores.softmax(dim=-1)
        if uniform:
            weights = torch.ones_like(weights) / inputs.size(1)
        context = torch.bmm(weights.unsqueeze(1), self.value(inputs)).squeeze(1)
        return self.classifier(context), weights


@torch.no_grad()
def evaluate(model, batch, uniform=False):
    model.eval()
    inputs, labels, selected = batch
    logits, weights = model(inputs, uniform=uniform)
    return {
        'loss': F.cross_entropy(logits, labels).item(),
        'accuracy': (logits.argmax(-1) == labels).float().mean().item(),
        'selected_mass': weights[torch.arange(len(labels)), selected].mean().item(),
    }


def main():
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
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1:
            print('scorer gradient norm:', model.scorer[0].weight.grad.norm().item())
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
