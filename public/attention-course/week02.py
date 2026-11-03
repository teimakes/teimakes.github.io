"""Week 2: train Q/K/V projections on associative retrieval."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def make_batch(size, generator, slots=5, key_count=12, classes=6):
    """이름에 연결된 값을 질문하는 검색 표본을 만듭니다.

    keys: (B, T). key_count개 이름에 무작위 점수를 부여해 정렬하고
        앞의 slots개를 선택합니다. 표본 안에서는 이름이 중복되지 않습니다.
    values: (B, T). 각 항목의 값은 classes개 클래스에서 독립적으로 뽑습니다.
    chosen: (B,). 질문할 항목의 위치를 고릅니다.
    rows: (B,). 각 표본의 행 인덱스입니다.
    keys[rows, chosen]: (B,). 질문할 이름 ID입니다.
    values[rows, chosen]: (B,). 그 이름에 연결된 정답 값입니다.

    반환한 chosen은 위치 검색 정확도에만 사용합니다. 학습은 값 정답으로
    수행하므로 모델이 어떤 항목을 참조해야 하는지 직접 감독하지 않습니다.
    """
    keys = torch.rand(size, key_count, generator=generator).argsort(-1)[:, :slots]
    values = torch.randint(classes, (size, slots), generator=generator)
    chosen = torch.randint(slots, (size,), generator=generator)
    rows = torch.arange(size)
    return keys, values, keys[rows, chosen], values[rows, chosen], chosen


class Retriever(nn.Module):
    def __init__(self, scale=True):
        """이름·값 임베딩과 Q·K·V 투영, 분류 계층을 만듭니다.

        key_embedding은 12개 이름을 각각 32차원으로 표현합니다.
        질문과 항목 이름은 이 임베딩을 공유합니다.
        value_embedding은 값 클래스 6개를 각각 32차원으로 표현합니다.
        wq와 wk는 이름 표현을 비교할 Query와 Key로 변환합니다.
        wv는 값 표현을 가중합할 Value로 변환합니다.
        readout은 문맥 벡터에서 여섯 클래스의 logits를 계산합니다.

        scale=False이면 점수를 sqrt(32)로 나누는 연산을 생략합니다.
        """
        super().__init__()
        self.scale = scale
        self.key_embedding = nn.Embedding(12, 32)
        self.value_embedding = nn.Embedding(6, 32)
        self.wq = nn.Linear(32, 32, bias=False)
        self.wk = nn.Linear(32, 32, bias=False)
        self.wv = nn.Linear(32, 32, bias=False)
        self.readout = nn.Linear(32, 6)

    def forward(self, keys, values, query, uniform=False):
        """질문과 항목의 Q·K 유사도로 V를 가중합해 값을 예측합니다.

        keys와 values는 (B, T), query는 (B,) 정수 ID입니다.

        q: 이름 임베딩에 wq를 적용하고 길이 1인 질문 축을 추가합니다.
           모양은 (B, 1, 32)입니다.
        k: 항목 이름에 wk를 적용한 (B, T, 32) 텐서입니다.
        v: 항목 값에 wv를 적용한 (B, T, 32) 텐서입니다.
        scores: q @ k.transpose(-2, -1)로 (B, 1, T)를 계산합니다.
           scale이 켜져 있으면 Key 차원의 제곱근으로 나눕니다.
        weights: 마지막 항목 축을 softmax로 정규화합니다.
           uniform=True이면 학습한 비율 대신 1/T를 사용합니다.
        context: weights @ v의 결과 (B, 1, 32)에서 질문 축만 제거합니다.
        readout: (B, 32)를 (B, 6) logits로 변환합니다.

        logits와 질문 축을 제거한 (B, T) weights를 반환합니다.
        """
        q = self.wq(self.key_embedding(query)).unsqueeze(1)
        k = self.wk(self.key_embedding(keys))
        v = self.wv(self.value_embedding(values))
        # 각 query와 key의 내적으로 위치별 유사도를 계산합니다.
        scores = q @ k.transpose(-2, -1)
        if self.scale:
            # 내적의 크기를 key 차원의 제곱근으로 조절합니다.
            scores = scores / math.sqrt(k.size(-1))
        # 위치별 점수를 확률로 바꾸어 어느 입력을 참고할지 정합니다.
        weights = scores.softmax(-1)
        if uniform:
            weights = torch.ones_like(weights) / keys.size(1)
        # 헤드별 문맥을 이어 붙여 원래 모델 차원으로 복원합니다.
        context = (weights @ v).squeeze(1)
        return self.readout(context), weights.squeeze(1)


@torch.no_grad()
def evaluate(model, batch, uniform=False):
    """값 예측과 정답 위치 검색을 구분해 평가합니다.

    loss는 값 클래스에 대한 교차 엔트로피입니다.
    accuracy는 여섯 값 중 예측한 값이 target과 일치한 비율입니다.
    retrieval_accuracy는 최대 가중치 위치가 chosen과 같은 비율입니다.
    서로 다른 항목의 값이 같을 수 있어 두 정확도는 다를 수 있습니다.

    평가 모드와 no_grad를 사용하며, uniform=True인 개입도 지원합니다.
    """
    model.eval()
    keys, values, query, target, chosen = batch
    logits, weights = model(keys, values, query, uniform)
    return {
        'loss': F.cross_entropy(logits, target).item(),
        'accuracy': (logits.argmax(-1) == target).float().mean().item(),
        'retrieval_accuracy': (weights.argmax(-1) == chosen).float().mean().item(),
    }


def main():
    """검색 모델을 학습하고 값 예측과 항목 선택을 평가합니다.

    --no-scale은 Q·K 점수 스케일링을 끕니다. 나머지 실행 설정은
    --steps, --seed, --out으로 지정합니다.
    검증 2,048개는 독립 생성기로 한 번 만들고 학습 내내 유지합니다.
    학습 배치 128개에서는 값 정답에 대한 교차 엔트로피만 사용합니다.
    AdamW 갱신 전 기울기를 비우고 역전파·기울기 클리핑을 수행합니다.
    100단계마다 값 정확도와 위치 검색 정확도를 함께 기록합니다.

    학습 후 이름과 값의 항목 순서를 같은 순열로 바꾸어도 logits가
    유지되는지 확인합니다. 가중치 합도 검사하고, 균등 가중치 개입과
    학습 이력을 metrics.json에, 모델 상태를 model.pt에 저장합니다.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=700)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-scale', action='store_true')
    parser.add_argument('--out', default='runs/week02')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    train_rng = torch.Generator().manual_seed(args.seed + 1)
    validation = make_batch(2048, torch.Generator().manual_seed(args.seed + 1000))
    model = Retriever(scale=not args.no_scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    history = [{'step': 0, **evaluate(model, validation)}]
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        keys, values, query, target, _ = make_batch(128, train_rng)
        logits, _ = model(keys, values, query)
        # 정답과 logits를 비교합니다. 소프트맥스는 손실 함수 내부에서 처리합니다.
        loss = F.cross_entropy(logits, target)
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
        keys, values, query, _, _ = validation
        before, weights = model(keys[:4], values[:4], query[:4])
        permutation = torch.tensor([4, 2, 0, 1, 3])
        after, _ = model(keys[:4, permutation], values[:4, permutation], query[:4])
        assert torch.allclose(before, after, atol=1e-5)
        assert torch.allclose(weights.sum(-1), torch.ones(4), atol=1e-6)
    report = {'torch': torch.__version__, 'seed': args.seed, 'scale': model.scale,
              'seconds': time.perf_counter() - started, 'history': history,
              'uniform_intervention': evaluate(model, validation, True)}
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    torch.save(model.state_dict(), output / 'model.pt')
    print('permutation invariance: passed')


if __name__ == '__main__':
    main()
