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
    """A·B의 값과 질문으로 문장을 만들고 숫자 쌍 단위로 분리합니다.

    a, b는 각각 0~5입니다. (a+b)%5가 0이면 validation, 1이면 test,
    나머지이면 train에 넣습니다. split과 다른 조합은 건너뜁니다.
    각 조합에서 A와 B 질문을 하나씩 만들고 해당 이름의 값을 답으로 둡니다.
    문장은 BOS A a B b ASK query ANS answer EOS의 열 토큰입니다.
    TO_ID로 정수화하고 (문장 수, 10) long 텐서를 반환합니다.

    같은 숫자 쌍의 두 질문이 서로 다른 분할로 들어가지 않도록 합니다.
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
        """인과적 멀티헤드 셀프 어텐션의 계층과 마스크를 등록합니다.

        qkv는 D차원에서 3D차원을 출력합니다. projection은 결합된 헤드를
        다시 D차원으로 변환합니다. Dh=D/heads를 사용합니다.
        future는 (context,context)이며 대각선 위의 위치만 True입니다.
        register_buffer로 등록한 마스크는 모델과 함께 저장·이동되지만
        optimizer가 갱신할 파라미터에는 포함되지 않습니다.
        """
        super().__init__()
        self.heads, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.projection = nn.Linear(dim, dim)
        # 마스크는 파라미터가 아닌 버퍼로 등록해 모델과 함께 저장합니다.
        self.register_buffer('future', torch.ones(context, context, dtype=torch.bool).triu(1))

    def forward(self, x):
        """입력 위치별로 인과적 멀티헤드 어텐션을 계산합니다.

        x: (B,T,D).
        qkv(x): (B,T,3D)를 (B,T,3,H,Dh)로 재배치합니다.
        permute(2,0,3,1,4): (3,B,H,T,Dh)로 축을 바꿉니다.
        unbind(0): Q·K·V를 각각 (B,H,T,Dh) 텐서로 분리합니다.
        scores: Q @ K.transpose(-2,-1) / sqrt(Dh), 모양은 (B,H,T,T)입니다.
        masked_fill: 현재 T에 해당하는 미래 마스크로 점수를 -inf로 바꿉니다.
        softmax(-1): 각 Query가 참조할 Key 위치의 가중치를 계산합니다.
        weights @ v: (B,H,T,Dh) 문맥을 계산합니다.
        transpose와 reshape: 헤드를 결합해 (B,T,D)로 복원합니다.
        projection: 헤드별 결과를 학습 가능한 출력 행렬로 변환합니다.

        입력과 같은 모양의 텐서를 반환합니다. 배치와 헤드 축은 서로
        독립적으로 계산하며 Query마다 자기 자신과 이전 위치만 참조합니다.
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
        """pre-norm 디코더 블록의 하위 계층을 만듭니다.

        ln1은 어텐션 입력, ln2는 FFN 입력을 정규화합니다.
        attention은 인과적 멀티헤드 셀프 어텐션입니다.
        ffn은 D→4D 선형 변환, GELU, 4D→D 선형 변환으로 구성합니다.
        모든 위치에 같은 FFN 가중치를 적용하며 위치 간 연산은 하지 않습니다.
        """
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = Attention(dim, heads, context)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        """어텐션과 FFN의 출력에 각각 잔차 연결을 적용합니다.

        ln1(x)를 어텐션에 입력하고 결과를 기존 x에 더합니다.
        갱신된 x에 ln2를 적용한 뒤 FFN을 계산하고 다시 x에 더합니다.
        정규화를 하위 계층 앞에 적용하므로 pre-norm 구조입니다.
        두 덧셈이 가능하도록 입력·어텐션·FFN의 마지막 차원은 모두 D입니다.
        """
        # 어텐션 출력에 입력을 더하는 첫 번째 잔차 연결입니다.
        x = x + self.attention(self.ln1(x))
        # FFN 출력에도 입력을 더해 두 번째 잔차 연결을 만듭니다.
        return x + self.ffn(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, dim=48, heads=4, layers=2, context=16):
        """토큰과 위치 임베딩, 디코더 블록, 어휘 출력 계층을 만듭니다.

        token은 12개 어휘를 D차원으로 표현합니다.
        position은 context개 위치를 D차원으로 표현합니다.
        blocks는 서로 다른 파라미터를 가진 디코더 블록 layers개입니다.
        norm은 마지막 블록 출력을 정규화합니다.
        head는 D차원을 어휘 수만큼의 logits로 변환합니다.

        기본값은 D=48, 헤드 4개, 블록 2개, 최대 문맥 16입니다.
        """
        super().__init__()
        self.context = context
        self.token = nn.Embedding(len(VOCAB), dim)
        self.position = nn.Embedding(context, dim)
        self.blocks = nn.Sequential(*[Block(dim, heads, context) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, len(VOCAB))

    def forward(self, ids):
        """토큰 ID를 모든 위치의 다음 토큰 logits로 변환합니다.

        ids는 (B,T) 정수 텐서이며 T가 context를 넘으면 예외를 발생시킵니다.
        positions는 0부터 T-1까지의 위치 ID입니다.
        token(ids)와 position(positions)를 더해 (B,T,D)를 만듭니다.
        위치 임베딩은 배치 축에 동일한 위치 번호로 적용됩니다.
        blocks에서 인과적 문맥을 계산하고 norm으로 최종 표현을 정규화합니다.
        head의 반환값은 (B,T,어휘 수) logits이며 확률로 정규화하지 않습니다.
        """
        if ids.size(1) > self.context:
            raise ValueError('input exceeds context length')
        positions = torch.arange(ids.size(1), device=ids.device)
        x = self.token(ids) + self.position(positions)
        return self.head(self.norm(self.blocks(x)))

    @torch.no_grad()
    def generate(self, ids, new_tokens=2, temperature=1.0, sample=False):
        """ANS까지의 입력 뒤에 토큰을 순차적으로 추가합니다.

        ids는 (B,T)이고 기본적으로 새 토큰 두 개를 생성합니다.
        eval()로 평가 모드를 설정하고 no_grad로 기울기를 기록하지 않습니다.
        입력이 최대 문맥 길이에 도달하면 생성을 중단합니다.
        self(ids)[:, -1]은 마지막 입력 뒤의 다음 토큰 logits입니다.
        양의 temperature로 나눈 뒤 sample=True이면 확률에서 표본을 뽑고,
        False이면 argmax로 점수가 가장 큰 ID를 선택합니다.
        cat(..., dim=1)로 선택한 ID를 입력 뒤에 추가하고 반복합니다.

        반환값에는 원래 프롬프트도 포함됩니다. EOS별 조기 종료는 없으므로
        이 실습에서는 답 숫자와 EOS에 해당하는 두 토큰만 생성합니다.
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
    """전체 다음 토큰 손실과 답 위치 정확도를 따로 계산합니다.

    data는 (문장 수,10)입니다. 앞의 9토큰은 입력, 뒤의 9토큰은 정답입니다.
    logits.reshape(-1,V)와 정답 reshape(-1)로 각 위치의 교차 엔트로피를
    계산하고 평균합니다. exp(loss)는 같은 토큰화 조건의 퍼플렉서티입니다.
    logits[:,7]은 ANS 위치의 출력이며 data[:,8]의 답 숫자와 비교합니다.
    accuracy는 이 답 숫자 위치만 평가하며 모든 위치의 정확도가 아닙니다.

    평가 모드에서 기울기 없이 계산한 세 지표를 반환합니다.
    """
    model.eval()
    logits = model(data[:, :-1])
    # 정답과 logits를 비교합니다. 소프트맥스는 손실 함수 내부에서 처리합니다.
    loss = F.cross_entropy(logits.reshape(-1, len(VOCAB)), data[:, 1:].reshape(-1))
    answer_accuracy = (logits[:, 7].argmax(-1) == data[:, 8]).float().mean()
    return {'loss': loss.item(), 'perplexity': loss.exp().item(),
            'accuracy': answer_accuracy.item()}


def main():
    """디코더 언어 모델을 학습하고 선택한 체크포인트로 테스트합니다.

    1. 숫자 쌍으로 분리한 train/validation/test를 생성하고 중복을 검사합니다.
    2. 모델과 AdamW를 만들고 학습 전 학습·검증 손실을 기록합니다.
    3. 학습 문장 64개를 복원 추출해 입력과 정답을 한 토큰 어긋나게 만듭니다.
    4. 모든 위치의 logits와 라벨로 교차 엔트로피를 계산하고 역전파합니다.
    5. 기울기 노름을 제한한 뒤 optimizer.step()으로 파라미터를 갱신합니다.
    6. 100단계마다 전체 학습·검증 문장으로 평가합니다. 검증 손실이
       개선될 때 모델 상태, 어휘, step을 best.pt에 저장합니다.
    7. best.pt를 다시 읽고 테스트 프롬프트에서 답 숫자와 EOS를 생성합니다.
    8. 미래 입력 변경에 대한 과거 출력 불변성과 모델 복원 결과를 검사합니다.
    9. 선택된 모델의 테스트 지표와 전체 이력을 metrics.json에 기록합니다.

    마지막 학습 단계의 모델과 검증으로 선택한 모델은 다를 수 있습니다.
    테스트 지표는 체크포인트 선택 조건으로 사용하지 않습니다.
    """
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
