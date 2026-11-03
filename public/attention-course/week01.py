"""Week 1: learn attention pooling from classification supervision."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def make_batch(size, generator, length=8):
    """표식 위치의 수치 부호를 분류하는 표본을 생성합니다.

    size는 배치 크기 B, length는 입력 위치 수 T입니다. generator를
    인자로 받아 학습 표본과 검증 표본의 난수 상태를 따로 관리합니다.

    values: (B, T). 각 위치의 수치를 표준 정규분포에서 뽑습니다.
    selected: (B,). 표본마다 정답을 결정할 위치 하나를 고릅니다.
    marker: (B, T). selected 위치만 1이고 나머지는 0인 표식입니다.
    inputs: (B, T, 2). 표식과 수치를 마지막 축으로 결합합니다.
    labels: (B,). 선택한 위치의 수치가 양수이면 1, 아니면 0입니다.

    inputs, labels, selected를 반환합니다. selected는 가중치를 분석할
    때 사용하며, 학습 손실에 위치 정답으로 전달하지 않습니다.
    """
    values = torch.randn(size, length, generator=generator)
    selected = torch.randint(length, (size,), generator=generator)
    marker = F.one_hot(selected, length).float()
    inputs = torch.stack([marker, values], dim=-1)
    labels = (values[torch.arange(size), selected] > 0).long()
    return inputs, labels, selected


class AttentionClassifier(nn.Module):
    def __init__(self):
        """위치 점수, 값 변환, 최종 분류를 담당하는 계층을 만듭니다.

        scorer의 첫 Linear는 위치별 두 특성을 16차원으로 변환합니다.
        Tanh 이후 두 번째 Linear가 위치마다 점수 하나를 출력합니다.
        value는 같은 두 특성을 가중합에 사용할 16차원 벡터로 만듭니다.
        classifier는 문맥 벡터를 두 클래스의 logits로 변환합니다.

        각 Linear의 가중치와 편향은 학습 중 함께 갱신됩니다.
        """
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
        self.value = nn.Linear(2, 16)
        self.classifier = nn.Linear(16, 2)

    def forward(self, inputs, uniform=False):
        """위치 가중치를 계산하고 입력을 분류합니다.

        inputs는 (B, T, 2)입니다. 마지막 축의 순서는 표식, 수치입니다.

        scores = scorer(inputs).squeeze(-1)
            (B, T, 1)에서 마지막 축만 없애 (B, T) 점수를 얻습니다.
        weights = scores.softmax(dim=-1)
            표본별 T개 위치의 합이 1이 되도록 정규화합니다.
        uniform=True
            평가 실험에서만 가중치를 1/T로 바꿉니다. 분류기와 value는
            그대로 사용하므로 균등 풀링 모델을 재학습하는 것과 다릅니다.
        context = bmm(weights.unsqueeze(1), value(inputs)).squeeze(1)
            (B, 1, T)와 (B, T, 16)의 곱으로 (B, 16)을 얻습니다.
            각 표본 안에서만 가중합하며 서로 다른 표본은 섞지 않습니다.
        classifier(context)
            문맥 벡터에서 (B, 2) logits를 계산합니다.

        반환값은 logits와 (B, T) weights입니다.
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
    """고정된 표본으로 분류 성능과 선택 가중치를 측정합니다.

    @torch.no_grad()는 기울기 기록을 중지합니다. model.eval()은
    평가 모드를 설정합니다. 학습을 다시 할 때는 train()이 필요합니다.

    cross_entropy는 (B, 2) logits와 (B,) 정수 라벨의 손실입니다.
    accuracy는 argmax로 고른 클래스가 라벨과 같은 표본의 비율입니다.
    selected_mass는 각 표본의 selected 위치에 준 가중치의 평균입니다.
    uniform을 forward로 전달하면 균등 가중치 개입도 같은 기준으로 평가합니다.

    item()으로 각 지표를 Python 숫자로 바꿔 사전에 담아 반환합니다.
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
    """표본 선택 분류기를 학습하고 가중치 변화를 기록합니다.

    1. --steps, --seed, --out을 읽고 CPU 스레드를 하나로 설정합니다.
    2. 모델 초기화 시드와 학습·검증 데이터의 생성기를 따로 정합니다.
    3. 검증 2,048개를 고정하고 학습 전 지표를 history에 기록합니다.
    4. 매 단계 새 학습 표본 128개로 분류 손실을 계산합니다.
    5. zero_grad → backward → 기울기 클리핑 → step 순서로 갱신합니다.
       첫 단계에는 scorer 기울기를 출력하고, 끝나면 파라미터 변화를 검사합니다.
    6. 100단계마다 검증하고 선택 위치의 평균 가중치를 기록합니다.
    7. 가중치 합이 1인지 검사하고 균등 가중치 개입의 성능을 계산합니다.
    8. metrics.json과 model.pt를 지정한 출력 폴더에 저장합니다.
    """
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
