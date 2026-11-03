# 전체 문맥과 BM25 RAG 비교 실험

가상의 가온 AI 클래스 안내서에 관한 같은 질문을 두 방식으로 풉니다.
2026-11-03에 gpt-6-luna로 연결 확인 4회와 본 실험 24회를 실행했습니다.

설명: https://teimakes.github.io/posts/full-context-vs-rag-ko/

## API 없이 결과 확인

```bash
git clone https://github.com/teimakes/teimakes.github.io.git
cd teimakes.github.io/public/qa-comparison
python analysis/inspect_example.py
python analysis/cache_costs.py
```

위 두 명령은 표준 라이브러리와 공개된 결과만 사용합니다. API 키와 네트워크 호출이 필요 없습니다.

## 파일

- `data/`: 7페이지 가상 문서, dev 12문항, 미실행 holdout 12문항.
- `retrieval.py`: 문단 ID 기준 청킹과 BM25 검색.
- `prompts.py`, `scoring.py`: 생성 지침, JSON 검증과 구조화 정답 채점.
- `experiment.py`: 계획, 계정 확인, 예산 예약, API 호출, 응답 기록.
- `plans/`, `budget-ledger.json`: 실제 실행 계획과 호출 예약 이력.
- `outputs/`: 질문·입력 문맥·생성 답변·사용량·시간·지표.
- `analysis/cache_costs.py`: 캐시 쓰기를 포함한 비용 후처리.
- `musique_adapter.py`: 후속 MuSiQue 실험을 위한 입력/정답 분리 어댑터.

공개 JSONL에는 원래 요청과 답변, 채점 정보가 보존돼 있습니다. `raw_response`는 상태, 모델, usage 필드로 축약했으며 서비스 응답 식별자와 계정 관련 메타데이터는 제거했습니다. 계정 설정과 API 키는 포함하지 않습니다.

`outputs/dev/summary.json`의 `token_cost_estimate_usd`는 캐시 쓰기 할증을 반영하기 전 계산입니다. 최종 비용에는 **`outputs/cache_costs.json`**을 사용하세요. 기본 30분 쓰기 가정의 전체 비용은 $0.00977476, 12시간 쓰기 단가를 적용한 민감도 계산은 $0.01817226입니다. 청구 확정액이 아닙니다.

사람이 평가하는 의미 정확도와 근거 충실성 칸은 비어 있습니다. ID F1과 의미적 충실성을 구별하세요.

## 코드 검증과 새 실행 계획

Python 3.11 이상입니다.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python experiment.py plan --limit 2 --out plans/new-smoke.json
```

`plan`은 API를 호출하지 않고 호출 수, 예약액, 승인용 해시를 출력합니다. 실제 실행한 코드와 데이터는 과거 계획의 해시로 확인할 수 있습니다.

새 유료 실험을 하려면 본인 계정의 모델 접근 권한, 단가, 크레딧과 실행 기한을 확인해 `config.json`과 별도의 `account.json`에 반영하고 계획을 다시 생성해야 합니다. 배포된 설정의 실행 차단일은 **2026-11-07 00:00 KST**입니다. 그 이후에는 자동으로 중단됩니다. `account.example.json`은 구조 예시이며 확인되지 않은 값은 채우지 않습니다. 실행기는 확인 시각이 24시간보다 오래됐어도 중단합니다.

인증값은 본인 실행 환경의 `OPENAI_API_KEY`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`로 설정합니다. 실행기는 `.env.local`을 자동으로 읽지 않습니다. 평문 키를 코드나 결과 파일에 넣지 마세요.

유료 실행은 검토한 계획의 해시를 직접 지정합니다.

```bash
python experiment.py run --plan plans/new-smoke.json --approve 검토한_계획의_해시 --account account.json --out outputs/new-smoke
```

공개된 원장은 원래 28회를 재전송하지 못하도록 유지됩니다. 계획이 같으면 `Call already attempted`로 중단되는 것이 정상입니다. 날짜나 모델 등 실제 조건을 바꾼 새 계획을 만들고 예산을 검토하세요. 과거 원장을 삭제해 재호출 제한을 우회하지 마세요.

로컬 상한 $0.25는 이 실행기의 예약액 합계를 제한합니다. 조직 전체의 강제 지출 한도는 아닙니다. 자동 재시도, 임베딩과 AI judge 호출은 없습니다.

MuSiQue 어댑터는 변환만 지원합니다. 현재 live CLI는 가온 문서 전용이며 MuSiQue의 문자열 답변 및 공식 평가기를 연결하는 작업은 후속 단계입니다.
