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
    """QUERY 다음에 첫 번째 기호를 복사하는 표본을 만듭니다.

    inputs는 (B, T)이며 0~7의 정수 기호를 무작위로 생성합니다.
    마지막 입력은 전용 QUERY 기호 ID인 8로 바꿉니다.
    labels는 같은 모양으로 만들고 처음에는 모두 -100을 넣습니다.
    QUERY 위치의 라벨만 첫 번째 입력 기호로 바꿉니다.

    cross_entropy는 -100 위치의 출력 손실을 무시합니다. 해당 위치가
    입력에서 제거되는 것은 아니며 QUERY가 그 표현을 참조할 수 있습니다.
    """
    inputs = torch.randint(8, (size, length), generator=generator)
    inputs[:, -1] = 8  # QUERY marker
    labels = torch.full_like(inputs, -100)
    labels[:, -1] = inputs[:, 0]  # answer after QUERY copies first symbol
    return inputs, labels


class CausalCopy(nn.Module):
    def __init__(self, context=12, dim=32):
        """지연 복사를 학습할 단일 헤드 셀프 어텐션을 구성합니다.

        token은 숫자 8종과 QUERY를 dim차원으로 표현합니다.
        position은 최대 context개 위치의 학습 가능한 임베딩입니다.
        norm은 각 토큰의 특성 축을 정규화합니다.
        qkv는 dim차원 입력을 3*dim으로 변환해 Q·K·V를 함께 계산합니다.
        output은 각 위치의 표현을 정답 숫자 8개의 logits로 바꿉니다.

        이 모델에는 별도의 FFN이나 여러 헤드를 추가하지 않습니다.
        """
        super().__init__()
        self.token = nn.Embedding(9, dim)
        self.position = nn.Embedding(context, dim)
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, 8)

    def forward(self, ids):
        """미래 위치를 차단하고 각 위치의 숫자 logits를 계산합니다.

        ids는 (B, T) 정수 입력입니다.
        x는 토큰 임베딩과 위치 임베딩을 더한 (B, T, D) 표현입니다.
        qkv(norm(x)).chunk(3, dim=-1)는 Q·K·V를 각각 (B, T, D)로 나눕니다.
        scores는 (B, T, T)이며 행은 Query, 열은 Key 위치입니다.
        triu(1)은 자기 자신을 제외한 미래 Key 위치만 True로 만듭니다.
        masked_fill은 그 점수를 -inf로 바꾸고 softmax로 가중치를 계산합니다.
        x + weights @ v는 원래 표현에 문맥을 더하는 잔차 연결입니다.
        output은 이를 (B, T, 8) logits로 변환합니다.

        logits와 (B, T, T) weights를 반환합니다. 학습 손실은 QUERY
        위치에만 적용하지만 다른 위치의 logits도 인과성 검사에 사용합니다.
        """
        length = ids.size(1)
        x = self.token(ids) + self.position(torch.arange(length, device=ids.device))
        # 정규화한 입력에서 Q·K·V를 각각 만듭니다.
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        # 각 query와 key의 내적으로 위치별 유사도를 계산합니다.
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        # 대각선 위쪽이 현재 위치에서 볼 수 없는 미래입니다.
        future = torch.ones(length, length, dtype=torch.bool, device=ids.device).triu(1)
        # 미래 위치를 소프트맥스 전에 가려 해당 가중치를 0으로 만듭니다.
        weights = scores.masked_fill(future, float('-inf')).softmax(-1)
        return self.output(x + weights @ v), weights


@torch.no_grad()
def evaluate(model, batch):
    """QUERY 답의 정확도와 첫 위치의 어텐션 가중치를 측정합니다.

    logits.flatten(0, 1)은 (B, T, 8)을 (B*T, 8)로 바꿉니다.
    y.flatten()은 (B*T,) 라벨이며 -100인 출력 위치는 손실에서 제외됩니다.
    accuracy는 마지막 위치가 첫 입력 기호를 맞혔는지 계산합니다.
    first_mass는 weights[:, -1, 0], 즉 QUERY가 첫 위치에 준 가중치의 평균입니다.

    반환 지표는 기울기가 없는 Python 숫자입니다.
    """
    model.eval()
    x, y = batch
    logits, weights = model(x)
    return {'loss': F.cross_entropy(logits.flatten(0, 1), y.flatten()).item(),
            'accuracy': (logits[:, -1].argmax(-1) == y[:, -1]).float().mean().item(),
            'first_mass': weights[:, -1, 0].mean().item()}


def main():
    """첫 기호 복사 모델을 학습하고 인과적 정보 범위를 검사합니다.

    --length는 학습 시퀀스 길이이며 위치 임베딩 범위인 2~12만 허용합니다.
    검증 표본을 고정하고 매 단계 새로운 학습 표본 128개를 생성합니다.
    logits의 배치·위치 축을 합친 뒤 -100이 아닌 라벨로 손실을 계산합니다.
    zero_grad, backward, 클리핑, AdamW 갱신 후 주기적으로 검증합니다.

    마지막 두 입력을 바꿔도 그 이전 logits가 유지되는지 검사합니다.
    미래 위치의 가중치가 0인지, 표본을 단독 또는 배치로 계산해도
    결과가 같은지도 확인합니다. 이 검사는 정답 정확도와 별개입니다.
    평가 이력과 예측 예시를 metrics.json, 모델 상태를 model.pt에 저장합니다.
    """
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
        # 정답과 logits를 비교합니다. 소프트맥스는 손실 함수 내부에서 처리합니다.
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
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
        x = validation[0][:4].clone()
        reference, weights = model(x)
        changed = x.clone()
        # 미래 토큰만 바꾸어 앞쪽 출력이 유지되는지 검사합니다.
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
