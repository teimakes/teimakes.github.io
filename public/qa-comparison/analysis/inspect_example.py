"""API 호출 없이 dev-02의 검색 결과와 저장된 답변을 확인합니다."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrieval import BM25, chunks


def inspect(case_id='dev-02'):
    """같은 질문의 검색 입력과 이미 생성된 두 답변을 나란히 출력합니다.

    handbook.txt는 두 방식이 공유하는 원문입니다. chunks()는 문단 ID를
    경계로 원문을 나눕니다. search()에는 question과 k만 전달하므로
    정답과 기준 근거 ID가 검색 결과에 영향을 줄 수 없습니다.

    기준 근거는 검색이 끝난 뒤 결과를 평가할 때만 사용합니다.
    마지막 반복문은 저장된 응답을 읽으며 새 답변을 생성하지 않습니다.
    """
    cases = [json.loads(s) for s in (ROOT / 'data/dev.jsonl').read_text().splitlines()]
    case = next(c for c in cases if c['id'] == case_id)
    document = (ROOT / 'data/handbook.txt').read_text()
    selected = BM25(chunks(document)).search(case['question'], k=5)
    retrieved = [p['id'] for p in selected]
    missing = sorted(set(case['sources']) - set(retrieved))
    print('질문:', case['question'])
    print('검색 문단:', ', '.join(retrieved))
    print('검색에서 빠진 기준 근거:', ', '.join(missing))
    for line in (ROOT / 'outputs/dev/results.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['id'] == case_id:
            print(row['arm'], json.dumps(row['parsed'], ensure_ascii=False))


if __name__ == '__main__':
    inspect()
