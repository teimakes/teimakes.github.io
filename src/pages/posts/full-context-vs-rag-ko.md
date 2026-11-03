---
layout: '../../layouts/Post.astro'
title: '문서를 전부 넣을까, 필요한 부분만 찾을까: 전체 문맥과 RAG 비교 실험'
description: '같은 문서와 12개 질문으로 전체 문맥 제공 방식과 BM25 RAG를 비교합니다. 정답, 근거, 토큰, 응답 시간과 비용을 측정하고 검색 누락이 답변에 미치는 영향을 살펴봅니다.'
date: '2026-10-31'
updatedDate: '2026-11-03'
tags: ['llm', 'rag', 'evaluation', 'python']
---

문서에 답이 있는 질문을 LLM에 맡길 때는 먼저 문서를 어떻게 전달할지 정해야 합니다. 짧은 안내서라면 원문을 통째로 넣을 수 있습니다. 문서가 많아지면 질문과 관련된 부분을 먼저 찾아서 넣는 방법도 생각해 볼 수 있습니다.

검색을 거치면 입력을 줄일 수 있지만, 답에 필요한 문단이 빠질 수 있습니다. 반대로 전체 문서를 제공하면 필요한 내용을 모두 전달할 수 있는 대신 질문마다 읽어야 할 입력이 늘어납니다. 실제 선택에는 정답률뿐 아니라 근거의 적절성, 응답 시간과 비용도 영향을 줍니다.

이 글에서는 **같은 문서와 같은 질문을 사용하고, 모델에 전달하는 문맥만 바꾸는 작은 실험**을 구현합니다. 가상의 교육 과정 안내서를 대상으로 전체 문맥 제공 방식과 BM25 검색을 사용하는 RAG를 비교합니다. 코드가 어떤 정보를 검색과 생성에 사용하고, 결과를 어떻게 채점하는지 설명한 뒤 실제 응답에서 두 방식의 차이를 확인합니다.

Python의 함수, 딕셔너리와 JSON 파일을 읽을 수 있으면 따라갈 수 있습니다. 공개된 응답을 분석하는 실습에는 API 키가 필요 없습니다. 모델을 새로 호출하는 과정은 별도로 구분했습니다. 아래 성능과 사용량은 **2026년 11월 3일에 실행한 결과**입니다.

## 목차

- [두 방식에서 달라지는 것](#approaches)
- [실험에 사용할 문서와 질문](#dataset)
- [같은 조건으로 비교하기](#design)
- [실험 코드의 전체 흐름](#implementation)
- [문단 분리와 BM25 검색](#code-chunks)
- [메시지 구성과 두 방식의 요청 만들기](#code-request)
- [실행 계획과 예산 관리](#code-plan)
- [API 응답 검증과 결과 저장](#code-response)
- [정답과 근거를 따로 평가하는 이유](#evaluation)
- [12문항의 실제 결과](#results)
- [답변이 달라진 할인 계산 문제](#failure)
- [토큰을 줄인 만큼 빨라졌을까?](#latency)
- [캐시를 반영한 API 비용](#cost)
- [직접 실행하고 확인하기](#practice)
- [MuSiQue와 파인튜닝으로 확장하기](#next)

<a id="approaches"></a>
## 두 방식에서 달라지는 것

**전체 문맥 제공 방식(full context)**은 질문을 보낼 때 안내서 전체를 함께 전달합니다. 모델이 답을 찾는 데 사용할 수 있는 범위가 원문 전체인 셈입니다. 이 글의 전체 문서는 모델의 입력 한도 안에 들어갑니다.

**RAG(Retrieval-Augmented Generation)**는 질문과 관련된 자료를 검색한 뒤 그 결과를 바탕으로 답변을 생성합니다. 여기서는 안내서를 문단으로 나누고 BM25로 상위 다섯 문단을 선택합니다. 생성 모델은 선택된 문단만 받습니다.

| 단계 | 전체 문맥 | 이번 실험의 RAG |
|---|---|---|
| 준비 | 안내서 원문 읽기 | 같은 원문을 문단으로 나눠 검색 인덱스 만들기 |
| 질문 처리 | 안내서 전체와 질문 전달 | 질문으로 문단을 검색하고 선택된 문단과 질문 전달 |
| 답변 생성 | 제공된 문서에서 답과 근거 찾기 | 제공된 검색 결과에서 답과 근거 찾기 |
| 확인할 실패 | 필요한 내용이 있어도 잘못 해석하거나 계산함 | 검색에서 근거를 놓치거나, 찾은 내용을 잘못 해석함 |

RAG에 벡터 데이터베이스나 임베딩 모델이 반드시 필요한 것은 아닙니다. 이번 구현은 단어 빈도를 이용하는 BM25를 검색기로 사용합니다. 임베딩, 재순위화, 질문 확장은 포함하지 않아 검색 단계에서 어떤 문단이 선택됐는지 직접 확인하기 쉽습니다.

파인튜닝은 이 두 방식과 실험 조건이 다릅니다. 학습 예제를 준비하고 모델의 파라미터를 갱신해야 하므로, 문맥 전달 방식만 바꾸는 첫 비교에서는 제외합니다. 학습 없이 만들 수 있는 기준 결과를 먼저 확보하는 것이 목적입니다.

<a id="dataset"></a>
## 실험에 사용할 문서와 질문

자료는 가상의 **가온 AI 클래스 안내서**입니다. 실제 교육 기관의 운영 규정이 아니라 실험용으로 작성한 문서이며, 가격·선수 조건·출석·과제·환불 관련 규정이 들어 있습니다. 원문은 7페이지 분량이고, A01부터 C12까지 총 31개의 문단 ID를 사용합니다.

문단마다 ID를 붙인 이유는 검색 결과와 답변의 근거를 연결하기 위해서입니다. 예를 들어 가격표는 A02, 할인 규정은 A10입니다. 모델이 “할인 규정에 따라 계산했다”고만 답하는 것보다 `sources: ["A02", "A10"]`처럼 근거를 반환하면 어떤 문단을 참조했는지 확인하기 쉽습니다.

질문은 개발용 12개와 별도로 보관한 확인용 12개로 나눴습니다. 이번에는 개발용 12개만 모델에 보냈습니다. 단순한 사실 조회뿐 아니라 조건 판단, 계산과 답변 보류도 포함했습니다.

| 질문 예시 | 확인할 능력 |
|---|---|
| G01의 정가는 몇 원인가? | 문서에서 수치 찾기 |
| 과거 수료자가 특정 날짜에 G02를 결제하면 얼마인가? | 가격과 할인 조건을 결합해 계산하기 |
| 정확히 90분 접속하면 출석인가? | ‘이상’에 경계값이 포함되는지 판단하기 |
| 과제 평균이 75점이면 수료 조건을 충족하는가? | 평균과 과제별 통과 기준을 구분하기 |
| 2027년 G02 수강료는 얼마인가? | 문서에 없는 정보를 추측하지 않기 |

각 질문에는 정답의 종류, 필요한 경우 수치 정답, 기준 답변과 근거 ID가 있습니다. 이 값들은 채점에만 사용합니다. 검색기와 생성 모델에 정답을 전달하면 어떤 시스템이 문서에서 답을 찾아냈는지 평가할 수 없기 때문입니다.

<a id="design"></a>
## 같은 조건으로 비교하기

두 방식은 같은 모델, 질문, 지침과 출력 형식을 사용합니다. 바뀌는 것은 사용자 메시지에 들어가는 문서 범위입니다.

| 항목 | 설정 |
|---|---|
| 실행일 | 2026-11-03 |
| 생성 모델 | `gpt-6-luna` |
| API / 서비스 등급 | Responses API / default |
| 추론 설정 | `reasoning.effort: none` |
| 최대 출력 | 요청당 1,024토큰 |
| RAG 청크 | 문단 ID 기준 31개 |
| 검색 | BM25, k1=1.5, b=0.75, top-k=5 |
| 본 실험 | 개발용 12문항 × 2방식 × 1회 = 24회 |
| 연결 확인 | 앞의 2문항 × 2방식 = 4회 |

문항마다 호출 순서도 바꿉니다. 첫 문항은 전체 문맥을 먼저 호출하고, 다음 문항은 RAG를 먼저 호출하는 식입니다. 어느 한 방식이 항상 먼저 실행되는 조건을 피하기 위한 조치입니다. 다만 이것만으로 네트워크나 서버 부하의 영향을 없앨 수는 없습니다.

연결 확인 4회를 마친 뒤에는 설정을 바꾸지 않고 본 실험을 실행했습니다. 전체 호출 수는 28회이며 모두 정상 응답과 사용량을 반환했습니다. 아래 정답률과 시간 비교에서는 연결 확인을 제외한 24회만 사용합니다.

생성 지침은 두 방식 모두 다음 내용을 포함합니다.

> 제공된 안내서만 근거로 답하고 조건, 기준 시점과 과정별 예외를 확인한다. 할인 계산에는 정가와 중복 적용 여부를 함께 확인한다. 필요한 정보가 부족하면 `unknown`으로 답하고 무엇을 더 확인해야 하는지 설명한다.

모델은 `answer`, `verdict`, `value`, `sources`를 가진 JSON을 반환합니다. `verdict`는 가능 여부에 따라 `yes` 또는 `no`, 수치 답변은 `value`, 문서만으로 판단할 수 없는 경우는 `unknown`입니다. 수치 질문의 `value`에는 단위를 제외한 숫자를 넣습니다.

이 형식을 고정하면 “십이만 원”과 “120,000원”처럼 표기가 다른 답변을 문자열 그대로 비교하는 문제를 줄일 수 있습니다. 대신 설명 전체가 정확한지는 별도의 검토가 필요합니다.

<a id="implementation"></a>
## 실험 코드의 전체 흐름

이 실험에서 가장 긴 파일은 310줄의 `experiment.py`입니다. 문서를 검색하고 답변을 받는 과정에 더해 실행 조건 검증, 예산 예약, 응답 파싱, 채점과 파일 저장을 담당합니다. 검색 자체는 41줄의 `retrieval.py`에 들어 있습니다. 두 파일을 나눠 읽으면 RAG의 계산과 실험을 관리하는 코드를 구분하기 쉽습니다.

| 파일 | 주요 함수·클래스 | 입력 → 결과 |
|---|---|---|
| `retrieval.py` | `chunks()`, `tokens()`, `BM25` | 원문과 질문 → 선택한 문단 |
| `prompts.py` | `messages()` | 문맥과 질문 → 모델에 보낼 메시지 |
| `scoring.py` | `Answer`, `check_fields()` | 생성 JSON과 기준 정답 → 검증된 답변과 정답 일치 여부 |
| `experiment.py` | `jobs()`, `plan()`, `run()` | 설정과 데이터 → 실행 계획 또는 실제 응답 기록 |
| `analysis/cache_costs.py` | `priced()`, `aggregate()` | 저장된 API 사용량 → 캐시를 반영한 비용 |
| `review.py` | 검토 CSV 집계 | 사람이 채운 검토 칸 → 검토율과 점수 |

`experiment.py`의 명령은 `plan`과 `run`으로 나뉩니다. **`plan`은 검색과 요청 구성을 수행하지만 API에 보내지 않습니다.** `run`은 저장된 계획을 검증한 다음 요청을 전송합니다. 두 명령이 같은 `jobs()`를 사용하므로 계획할 때와 실행할 때 문맥을 만드는 규칙이 같습니다.

실행 순서는 다음과 같습니다.

1. 설정과 질문을 읽고 문단 검색 인덱스를 만듭니다.
2. 질문마다 `full`과 `rag`의 요청을 만듭니다.
3. `plan`에서는 요청별 해시와 비용 예약액을 저장합니다.
4. `run`에서는 계획·계정·예산을 확인하고 요청을 보냅니다.
5. 응답에서 사용량과 답변을 꺼내 검증하고 채점합니다.
6. 문항별 기록을 저장한 뒤 방식별 결과를 집계합니다.

아래 코드는 공개된 소스의 주요 함수를 따라갑니다. 읽기 쉽도록 줄을 나누고 주석과 docstring을 덧붙였으며, 계산과 조건은 원래 함수와 같습니다. 함수 블록은 전체 파일의 일부입니다. 실제 실행에는 블록을 각각 복사하기보다 [공개된 실험 폴더](https://github.com/teimakes/teimakes.github.io/tree/main/public/qa-comparison)를 사용하세요.

### 1. 질문 한 개에는 어떤 정보가 들어 있을까?

`data/dev.jsonl`은 한 줄에 JSON 객체 하나를 저장합니다. 첫 질문은 다음과 같습니다.

```json
{
  "id": "dev-01",
  "category": "numeric",
  "question": "G01 정가를 원 단위 숫자로 알려주세요.",
  "expected": {"verdict": "value", "value": 120000},
  "reference_answer": "G01 전체 수강료는 부가세 포함 120,000원입니다.",
  "sources": ["A02"]
}
```

`id`는 같은 질문의 두 답변을 연결하는 키입니다. `question`은 검색기와 생성 모델이 받는 질문입니다. `expected`는 기계적으로 비교할 정답이고, `reference_answer`는 사람이 답변을 검토할 때 읽는 기준 문장입니다. `sources`는 기준 근거 ID입니다.

실행기 내부에서는 이 객체 전체를 `case`로 보관합니다. 하지만 API에 보내는 `request`를 만들 때는 `case['question']`만 꺼냅니다. 기준 정답이 실행기의 메모리에 있다는 사실과 모델의 입력에 들어간다는 사실은 다릅니다. 아래 `jobs()`와 `payload()`를 읽을 때 이 경계를 확인할 수 있습니다.

`load_cases(split)`은 `dev` 또는 `holdout` 파일을 읽고 빈 줄을 제외한 각 줄에 `json.loads()`를 적용합니다. `ROOT`는 `experiment.py`가 있는 폴더의 절대 경로이므로, 문서와 질문은 실행 명령을 입력한 위치가 아니라 코드 파일의 위치를 기준으로 찾습니다.

<a id="code-chunks"></a>
### 2. `chunks()`와 `tokens()`: 원문을 검색 가능한 단위로 바꾸기

`chunks()`는 문단 ID가 시작되는 위치를 찾고, 그 위치부터 다음 ID 직전까지를 하나의 청크로 묶습니다. 다음 두 함수에는 `re` 모듈을 사용합니다.

```python
def chunks(document):
    """문단 ID와 그 문단의 텍스트를 함께 반환합니다.

    ^는 줄의 시작, [ABC]는 문단 종류, 두 자리 숫자는 문단 번호입니다.
    re.M을 지정해 문서 전체에서 각 줄의 시작을 검사합니다.
    한 문단의 끝은 다음 ID의 시작이며 마지막 문단은 원문 끝까지입니다.
    """
    starts = list(re.finditer(r'^([ABC]\d{2}) ', document, re.M))
    if not starts:
        raise ValueError('No paragraph IDs found')
    return [
        {
            'id': m.group(1),
            'text': document[
                m.start():starts[i + 1].start()
                if i + 1 < len(starts) else len(document)
            ].strip(),
        }
        for i, m in enumerate(starts)
    ]


def tokens(text):
    """BM25에 사용할 단어와 한글 어절 내부의 두 글자 조합을 만듭니다.

    영문은 소문자로 통일하고 영문·숫자 묶음과 한글 어절을 추출합니다.
    한글 어절에는 연속한 두 글자를 추가해 부분적인 일치도 반영합니다.
    반환값은 생성 모델의 과금 토큰과 관계없는 검색용 문자열 목록입니다.
    """
    words = re.findall(r'[a-z0-9]+|[가-힣]+', text.lower())
    return words + [
        w[i:i + 2]
        for w in words
        if re.fullmatch('[가-힣]+', w)
        for i in range(len(w) - 1)
    ]
```

청크에는 `id`와 `text`를 따로 저장하지만 `text`에도 A02 같은 ID가 남습니다. 생성 모델이 해당 문단을 인용할 때 사용할 수 있도록 원문에서 제거하지 않은 것입니다. 원문 맨 앞의 제목처럼 첫 ID 이전의 내용은 RAG 청크에 포함되지 않습니다. 문단 사이에 있는 페이지 표시는 해당 청크에 남습니다. 전체 문맥 방식에는 원문을 그대로 전달합니다.

`tokens('G02 수강료')`의 결과는 `['g02', '수강료', '수강', '강료']`입니다. 문단과 질문에 같은 함수를 적용하므로 같은 단위로 비교할 수 있습니다. 형태소 분석은 하지 않으며, 예를 들어 의미만 비슷하고 문자열이 다른 표현까지 연결해 주는 방법은 아닙니다.

### 3. `BM25`: 문단 점수를 계산하고 다섯 개 선택하기

`BM25.__init__()`는 매 질문마다 다시 계산할 필요가 없는 문단 통계를 저장합니다. `Counter`, `math`와 앞의 `tokens()`를 사용합니다.

```python
class BM25:
    """문단별 빈도와 문단 길이를 이용하는 로컬 검색기입니다."""

    def __init__(self, paragraphs):
        """문단을 보관하고 검색 점수 계산에 필요한 통계를 만듭니다.

        counts[i]는 i번째 문단에서 각 검색 토큰이 등장한 횟수입니다.
        lengths는 문단별 검색 토큰 수, avg는 그 평균입니다.
        df는 각 토큰이 등장한 문단 수입니다. 한 문단에서 같은 토큰이
        여러 번 나와도 df에는 한 번만 반영됩니다.
        """
        self.paragraphs = paragraphs
        self.counts = [Counter(tokens(p['text'])) for p in paragraphs]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg = sum(self.lengths) / len(self.lengths)
        self.df = Counter(t for c in self.counts for t in c)

    def search(self, question, k=5):
        """질문과 각 문단의 점수를 계산해 상위 k개를 선택합니다.

        query는 집합이므로 질문 속 같은 토큰의 반복 횟수는 무시합니다.
        f는 현재 문단의 토큰 빈도입니다. 토큰이 없는 문단에서는 그
        토큰의 점수를 더하지 않습니다. 점수 동률은 원문 순서로 풉니다.
        선택이 끝나면 선택된 문단을 다시 원문 순서로 반환합니다.
        """
        query = set(tokens(question))
        scores = []
        for i, c in enumerate(self.counts):
            score = 0.0
            for t in query:
                f = c[t]
                if f:
                    idf = math.log(
                        1 + (len(self.counts) - self.df[t] + 0.5)
                        / (self.df[t] + 0.5)
                    )
                    score += idf * f * 2.5 / (
                        f + 1.5 * (
                            0.25 + 0.75 * self.lengths[i] / self.avg
                        )
                    )
            scores.append((i, score))

        chosen = sorted(scores, key=lambda x: (-x[1], x[0]))[:k]
        return [
            dict(self.paragraphs[i], retrieval_score=s)
            for i, s in sorted(chosen)
        ]
```

`df`와 `counts`는 역할이 다릅니다. ‘할인’이 한 문단에서 다섯 번 등장했다면 그 문단의 `counts['할인']`은 5입니다. 하지만 그 문단이 `df['할인']`에 더하는 값은 1입니다. 앞의 값은 해당 문단에서 얼마나 반복됐는지, 뒤의 값은 전체 문단에 얼마나 널리 퍼져 있는지를 나타냅니다.

`idf`는 많은 문단에 공통으로 나타나는 표현의 비중을 낮춥니다. 빈도 항은 같은 토큰이 반복될 때 점수를 높이되 증가 폭이 점차 작아지게 하고, `self.lengths[i] / self.avg`는 문단 길이 차이를 반영합니다. 코드의 1.5는 k1, 0.75는 b이고, 2.5는 k1+1, 0.25는 1−b입니다.

마지막 두 정렬도 구분해야 합니다. 첫 정렬은 `(-점수, 원문 위치)`를 사용해 검색할 문단을 선택합니다. 두 번째 정렬은 이미 선택한 문단의 원문 위치를 기준으로 합니다. 모델에 제공하는 순서를 검색 점수순으로 바꾸지 않기 위한 처리입니다.

이 구현에는 최소 점수 기준이 없습니다. 모든 문단의 점수가 낮더라도 최대 k개를 반환합니다. 또 `set()`으로 만든 질문 토큰의 순회 순서는 프로세스에 따라 달라질 수 있어 부동소수점 덧셈의 극히 작은 차이까지 항상 같다고 보장하지는 않습니다. 실제 결과의 `provided_ids`를 저장하는 이유 중 하나입니다. 큰 문서 집합에 적용하려면 검색 효율과 순위 안정성도 별도로 검토해야 합니다.

<a id="code-request"></a>
### 4. `messages()`와 `payload()`: 모델에 보내는 경계 확인하기

검색기가 반환한 문단을 받으면 `messages()`가 지침과 사용자 입력을 만듭니다. `PROMPTS`에는 `baseline`과 `grounded`가 있지만, 이번 실행은 두 방식 모두 `grounded`를 사용합니다.

```python
def messages(document, question, variant):
    """지침과 문서·질문을 두 메시지로 나눕니다.

    system에는 선택한 지침을, user에는 문서와 질문을 넣습니다.
    expected, reference_answer와 기준 sources를 받는 인자는 없습니다.
    document 태그는 텍스트 경계 표시이며 별도의 보안 장치는 아닙니다.
    """
    return [
        {"role": "system", "content": PROMPTS[variant]},
        {
            "role": "user",
            "content": f"<document>\n{document}\n</document>\n질문: {question}",
        },
    ]


def payload(c, document, question):
    """같은 모델 설정과 출력 스키마를 사용한 API 요청 본문을 만듭니다.

    c는 config.json에서 읽은 설정입니다. 두 방식의 차이는 document에
    들어가는 텍스트뿐입니다. SCHEMA는 experiment.py 상단의 JSON 스키마로
    answer, verdict, value, sources 네 필드와 각 필드의 자료형을 지정합니다.
    """
    return {
        'model': c['model'],
        'input': messages(document, question, 'grounded'),
        'reasoning': {'effort': c['reasoning_effort']},
        'max_output_tokens': c['max_output_tokens'],
        'store': False,
        'service_tier': 'default',
        'text': {
            'format': {
                'type': 'json_schema',
                'name': 'qa_answer',
                'strict': True,
                'schema': SCHEMA,
            }
        },
    }
```

`SCHEMA`는 네 필드를 필수로 요구하고 추가 필드를 허용하지 않습니다. `verdict`의 값은 정해진 네 종류로 제한하고 `value`는 숫자 또는 null로 지정합니다. 생성 형식을 제한하는 것만으로 답의 정확성이 보장되지는 않습니다. 예를 들어 형식에 맞는 `value: 180000`도 정답이 153000인 질문에서는 오답입니다.

인증값은 `payload()`가 반환하는 딕셔너리에 들어가지 않습니다. 나중에 `complete()`가 HTTP 헤더를 만들 때 환경 변수에서 읽습니다. 따라서 저장된 `request`에서는 모델이 실제로 받은 문맥을 확인할 수 있습니다.

<a id="code-jobs"></a>
### 5. `jobs()`: 질문 하나를 두 방식의 요청으로 펼치기

`jobs()`는 `yield`를 사용하는 제너레이터입니다. 요청을 하나씩 반환하지만 직접 전송하지 않습니다. 아래는 이 함수 전체입니다.

```python
def jobs(c, split, limit, repeats):
    """문서와 질문을 읽어 full/rag 요청을 순서대로 반환합니다.

    인덱스는 이 제너레이터를 순회하기 시작할 때 한 번 만듭니다.
    limit=0이면 질문 전체를 사용하고, 양수이면 앞의 limit개만 사용합니다.
    repeat는 같은 질문을 몇 번째 반복하는지 나타내는 0부터의 인덱스입니다.

    반환 객체의 case는 로컬 채점을 위한 자료입니다. API 전송 대상은
    request뿐이며, 그 안에는 선택한 문맥과 question만 들어갑니다.
    provided_ids는 실제 제공한 문단을 나중에 확인하기 위한 기록입니다.
    """
    document = (ROOT / 'data/handbook.txt').read_text(encoding='utf-8')
    paragraphs = chunks(document)
    index_start = time.perf_counter()
    index = BM25(paragraphs)
    index_s = time.perf_counter() - index_start
    cases = load_cases(split)
    if limit:
        cases = cases[:limit]

    for repeat in range(repeats):
        for i, case in enumerate(cases):
            start = time.perf_counter()
            selected = index.search(case['question'], c['top_k'])
            retrieval_s = time.perf_counter() - start
            arms = ['full', 'rag'] if (i + repeat) % 2 == 0 else ['rag', 'full']
            for arm in arms:
                context = document if arm == 'full' else '\n\n'.join(
                    p['text'] for p in selected
                )
                yield {
                    'key': f"{case['id']}:{repeat}:{arm}",
                    'arm': arm,
                    'repeat': repeat,
                    'case': case,
                    'provided_ids': [
                        p['id'] for p in (paragraphs if arm == 'full' else selected)
                    ],
                    'retrieval_s': retrieval_s if arm == 'rag' else 0.0,
                    'index_build_s': index_s if arm == 'rag' else 0.0,
                    'request': payload(c, context, case['question']),
                }
```

`arm`은 비교할 방식의 이름으로, 여기서는 `full` 또는 `rag`입니다. `key`는 질문 ID, 반복 번호와 방식을 결합한 문자열입니다. 첫 질문의 전체 문맥 요청은 `dev-01:0:full`이 됩니다. 같은 질문도 방식이나 반복 번호가 다르면 별도 요청으로 구분할 수 있습니다.

`(i + repeat) % 2`가 0인지 1인지에 따라 두 방식의 순서를 교대합니다. 무작위로 섞지 않고 같은 조건에서 정해진 순서를 사용하는 방법입니다. `repeats=1`이면 질문마다 먼저 호출할 방식이 바뀝니다.

`plan()`은 이 반복문을 끝까지 실행해 모든 요청을 확인하고, `run()`도 같은 과정을 별도로 수행합니다. 인덱스는 프로젝트 전체에서 딱 한 번 만드는 것이 아니라 **`jobs()`를 한 번 순회할 때마다 한 번** 만듭니다. 계획을 재검증할 때도 검색은 수행하지만 모델 API를 호출하지는 않습니다.

<a id="code-plan"></a>
### 6. `plan()`과 `verify_plan()`: 실행 조건을 기록하고 대조하기

`plan()`은 `jobs()`가 만든 요청마다 내용의 해시, 입력 추정량과 비용 예약액을 기록합니다. 검색 시간처럼 실행할 때마다 달라지는 측정값은 계획에 넣지 않으므로, 같은 설정과 소스에서 계획을 다시 만들어 비교할 수 있습니다.

```python
def plan(c, split='dev', limit=2, repeats=1):
    """전송 전에 모든 요청을 구성하고 건수, 예산과 재현 조건을 저장합니다.

    work에는 로컬에서 만든 요청이 들어가며 HTTP 통신은 수행하지 않습니다.
    calls에는 요청 본문 대신 해시와 비용 추정치를 남깁니다.
    files는 실행용 Python과 데이터 파일의 해시입니다.
    plan_sha256은 마지막에 붙이므로 자기 자신을 포함해 해시하지 않습니다.
    """
    validate_config(c)
    if limit < 0 or repeats < 1:
        raise ValueError('limit >=0, repeats >=1')
    work = list(jobs(c, split, limit, repeats))
    if not work:
        raise ValueError('Empty dataset')
    calls = [
        {
            'key': j['key'],
            'request_sha256': sha(j['request']),
            'input_byte_bound': input_bound(j['request']),
            'reserved_usd': reserve(j['request'], c),
        }
        for j in work
    ]
    obj = {
        'version': 1,
        'config': c,
        'split': split,
        'limit': limit,
        'repeats': repeats,
        'calls': calls,
        'max_calls': len(calls),
        'reserved_usd': sum(x['reserved_usd'] for x in calls),
        'files': fingerprints(),
        'status': 'AWAITING_APPROVAL_NO_LIVE_RESULTS',
    }
    obj['plan_sha256'] = sha(obj)
    return obj


def verify_plan(p, approval):
    """지정한 계획을 현재 코드에서 다시 만든 계획과 대조합니다."""
    if approval != p['plan_sha256']:
        raise ValueError('Explicit approval hash required')
    actual = plan(p['config'], p['split'], p['limit'], p['repeats'])
    if actual != p:
        raise ValueError('Plan/data/code changed; regenerate and obtain approval again')
    if p['reserved_usd'] > p['config']['project_cap_usd']:
        raise ValueError('Plan exceeds project cap')
```

`sha()`는 JSON의 키를 정렬하고 불필요한 공백을 제거한 형식으로 변환한 뒤 SHA-256을 계산합니다. 딕셔너리에 값을 넣은 순서만 달라졌을 때 해시가 달라지는 일을 피합니다. `fingerprints()`는 실험 폴더 바로 아래의 Python 파일과 `data/`의 TXT·JSONL을 대상으로 합니다. README와 분석용 하위 폴더까지 모든 파일을 검사하는 것은 아닙니다.

`--approve`에는 검토한 계획의 해시를 지정합니다. 해시를 안다는 사실 자체가 사람의 승인을 증명하지는 않습니다. 실행할 계획을 명시해서 다른 계획을 잘못 실행하는 일을 줄이는 확인 절차입니다. 원래 계획 JSON에 남은 `AWAITING_APPROVAL_NO_LIVE_RESULTS`는 계획을 만든 시점의 상태이며, 이후 실행했는지는 `outputs/`와 원장에서 확인합니다.

입력 예약량은 요청 JSON의 UTF-8 바이트 수에 1,024를 더해 구합니다. 실제 토크나이저로 센 값이나 API가 보장하는 상한은 아닙니다. 입력 예약 단가는 100만 토큰당 0.25달러, 출력은 설정한 최대 1,024토큰으로 계산합니다. 실행 뒤 보고된 사용량이 이 추정 조건을 넘으면 중단하지만, 이미 보낸 요청의 비용을 취소하는 장치는 아닙니다.

<a id="code-budget"></a>
### 7. `account_gate()`와 `Ledger`: 중단할 조건을 정하기

`account_gate()`는 실행 기한, 확인한 잔액, 모델 이용 가능 여부와 환경 변수를 검사합니다. 계정 확인 시각이 24시간보다 오래됐거나 조직·프로젝트가 설정과 다르면 중단합니다. **이 함수가 잔액 API나 모델 목록을 직접 조회하지는 않습니다.** 별도로 확인한 결과를 `account.json`에 기록하고, 그 값과 유효 기간을 검사합니다.

비용 예약은 `Ledger`가 담당합니다. 다음은 `__enter__()`에서 원장을 연 뒤 호출하는 메서드입니다.

```python
def reserve(self, plan_hash, key, usd):
    """요청을 보내기 전에 한 번의 호출에 필요한 예약액을 기록합니다.

    계획 해시와 요청 키의 조합으로 이미 시도한 요청인지 판별합니다.
    합계가 상한을 넘으면 HTTP 전송 전에 예외를 발생시킵니다.
    임시 파일을 끝까지 쓴 뒤 교체하므로, 기록 중인 JSON이 원장을
    덮어쓰지 않습니다. 호출 후 예약액을 돌려주는 처리는 없습니다.
    """
    identity = plan_hash + ':' + key
    if any(x['identity'] == identity for x in self.data['calls']):
        raise ValueError('Call already attempted; no automatic replay')
    if self.data['reserved_usd'] + usd > self.cap + 1e-12:
        raise ValueError('Project budget exhausted')
    self.data['reserved_usd'] += usd
    self.data['calls'].append({'identity': identity, 'reserved_usd': usd})
    temp = self.path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8') as f:
        f.write(dump(self.data))
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, self.path)
```

`__enter__()`는 `O_CREAT | O_EXCL`로 잠금 파일을 만듭니다. 두 실행이 같은 원장에 동시에 비용을 예약하지 않도록 하기 위해서입니다. 일반적인 종료에서는 `__exit__()`가 잠금 파일을 지웁니다. 강제 종료 등으로 파일이 남았다면 다른 실행이 실제로 동작 중인지와 원장 내용을 확인한 뒤 복구해야 합니다.

예약액을 실제 청구액으로 바꾸지는 않습니다. 성공한 요청도 최대 출력을 가정한 예약이 남기 때문에 실측 비용보다 먼저 로컬 예산이 소진될 수 있습니다. 이번 원장의 예약액이 약 0.10649달러이고 토큰 비용 추정치가 약 0.00977달러인 이유입니다.

이 방법은 과금 여부가 불확실한 요청을 이미 지출했을 가능성이 있는 것으로 취급합니다. 같은 원장을 사용하는 실행만 제어하므로 조직 전체의 지출이나 다른 프로그램의 호출을 제한하지는 않습니다.

<a id="code-response"></a>
### 8. `complete()`와 `parse_response()`: 통신과 답변 읽기를 나누기

모델에 요청을 보내는 함수는 `complete()`입니다. 다른 처리와 통신을 분리했기 때문에 오프라인 검사에서는 이 함수를 고정 응답으로 대체할 수 있습니다.

```python
def complete(request):
    """요청을 한 번 POST하고 응답 JSON과 소요 시간을 반환합니다.

    인증값은 환경 변수에서 읽어 HTTP 헤더에 넣습니다.
    타이머는 요청 객체를 만든 뒤 시작하고 응답 본문을 JSON으로
    읽을 때까지 측정합니다. 스트리밍하지 않으므로 첫 토큰이 나타나는
    대기 시간을 뜻하지 않습니다. 실패해도 다시 전송하지 않습니다.
    """
    headers = {
        'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY'],
        'Content-Type': 'application/json',
        'OpenAI-Organization': os.environ['OPENAI_ORG_ID'],
        'OpenAI-Project': os.environ['OPENAI_PROJECT_ID'],
    }
    req = urllib.request.Request(
        ENDPOINT, dump(request).encode(), headers=headers, method='POST'
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            data = json.load(response)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code}; stopped; no retry') from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError('Network/timeout; billing uncertain; stopped; no retry') from None
    return data, time.perf_counter() - start


def parse_response(data):
    """완료된 API 응답에서 출력 텍스트를 꺼내 검증합니다.

    output에는 여러 요소가 들어갈 수 있으므로 message 형식만 고릅니다.
    그 안의 content에서 output_text를 찾아 연결합니다.
    Answer로 변환할 수 없는 JSON이나 모순된 값은 예외로 처리합니다.
    """
    if data.get('status') != 'completed':
        raise ValueError('Incomplete/refused response')
    text = ''.join(
        c['text']
        for o in data.get('output', [])
        if o.get('type') == 'message'
        for c in o.get('content', [])
        if c.get('type') == 'output_text'
    )
    return Answer.model_validate_json(text), text
```

120초는 `urlopen()`에 전달하는 소켓 작업의 타임아웃입니다. 호출부터 종료까지를 항상 120초 이내로 제한하는 엄밀한 전체 시간 제한은 아닙니다. HTTP 오류나 타임아웃이 발생하면 실행기는 예외의 종류를 기록합니다. 상세 응답 본문과 인증 헤더를 오류 기록에 복사하지 않습니다.

`parse_response()`가 사용하는 `Answer`는 Pydantic 모델입니다. 문자열과 숫자 등 자료형을 엄격하게 검사하고 예상하지 않은 필드를 금지합니다. 또한 `verdict='value'`이면 `value`가 유한한 숫자여야 하고, 그 외에는 null이어야 한다는 일관성도 검사합니다. Python은 참·거짓 값을 정수처럼 다루는 경우가 있으므로 수치 답변에 bool을 넣는 것도 명시적으로 거부합니다.

여기서 검증하는 것은 답변의 형식입니다. `sources`가 실제 문서에 존재하는지와 답이 맞는지는 다음 채점 단계에서 확인합니다.

<a id="code-run"></a>
### 9. `run()`: 요청 한 건이 기록으로 남기까지

`run()`은 앞에서 살펴본 함수들을 연결합니다. 시작할 때 `verify_plan()`과 `account_gate()`를 호출하고, `exist_ok=False`로 출력 폴더를 만듭니다. 같은 결과 폴더를 덮어쓰지 않도록 하는 설정입니다. 계획, 실행 시작 시각과 라이브러리 버전은 `manifest.json`에 저장합니다.

반복문은 `jobs()`가 만든 요청을 계획 안의 `calls`와 순서대로 묶습니다. 각 요청은 계정 검사와 비용 예약을 마친 뒤 전송합니다. 다음은 `row`에 질문, 방식, 기준 정답과 제공 문단 ID를 넣은 뒤 수행하는 실제 처리 부분입니다.

```python
# run()의 문항별 처리에서 발췌했습니다. row, job, case, p 등은 앞에서 준비합니다.
started = time.perf_counter()
try:
    data, latency = complete(job['request'])
    row.update(
        raw_response=data,
        api_latency_s=latency,
        resolved_model=data.get('model'),
        response_id=data.get('id'),
    )
    row.update(usage_cost(data['usage'], p['config']))
    if (
        row['input_tokens'] > planned['input_byte_bound']
        or row['output_tokens'] > p['config']['max_output_tokens']
    ):
        raise ValueError('Usage exceeded reservation assumptions; stop')

    answer, text = parse_response(data)
    row.update(parsed=answer.model_dump(), text=text)
    all_ids = set(
        p['id']
        for p in chunks((ROOT / 'data/handbook.txt').read_text(encoding='utf-8'))
    )
    row.update(
        check_fields(answer, case, all_ids),
        **evidence(answer, case, job['provided_ids']),
    )
    row['status'] = 'ok'
except Exception as e:
    row['error_type'] = type(e).__name__

row['e2e_s'] = time.perf_counter() - started + job['retrieval_s']
rows.append(row)
```

응답이 도착하면 답변을 파싱하기 전에 `usage_cost()`로 사용량을 기록합니다. 답변 JSON의 형식이 틀려도 API는 이미 생성을 수행했을 수 있습니다. 이 순서로 처리하면 답변 검증에 실패한 요청의 사용량도 남길 수 있습니다.

`status='ok'`는 통신, 형식 검증과 채점 처리가 정상적으로 끝났다는 뜻입니다. **정답을 맞혔다는 뜻은 아닙니다.** 형식에 맞는 `unknown`을 반환해 오답 처리된 `dev-02`의 RAG도 상태는 `ok`, `structured_match`는 0입니다. 오답이어도 비교는 계속합니다. 통신이나 파싱 등에서 예외가 발생하면 해당 요청을 기록한 뒤 후속 호출을 중단합니다.

`e2e_s`에는 HTTP 요청 준비, 통신, 사용량 처리, 파싱, 채점과 검색 시간이 들어갑니다. 인덱스 생성, 계획 검증, 비용 원장 기록과 결과 파일 저장 시간은 포함하지 않습니다. 프로그램을 시작해서 모든 파일을 저장할 때까지 걸리는 전체 실행 시간과 구분해야 합니다.

### 10. `results.jsonl`과 `summarize()`: 나중에 검증할 수 있는 기록 남기기

요청 한 건의 처리가 끝나면 `row`를 JSON 한 줄로 만들어 `results.jsonl`에 추가하고 `flush()`와 `os.fsync()`로 기록을 반영합니다. 매번 `summary.json`도 갱신하므로 중간에 멈춰도 완료된 결과와 미실행 건수를 확인할 수 있습니다. 자동 재개나 실패한 요청의 자동 재전송은 하지 않습니다.

| 기록 필드 | 확인할 수 있는 내용 |
|---|---|
| `id`, `arm`, `repeat` | 어떤 질문, 방식과 반복의 결과인가 |
| `request`, `provided_ids` | 어떤 설정과 문맥을 모델에 전달했는가 |
| `parsed`, `text` | 검증한 JSON과 생성된 문자열 |
| `expected`, `reference_sources` | 무엇을 기준으로 채점했는가 |
| `structured_match`, `citation_f1` | 정답과 근거 ID의 일치 정도 |
| `input_tokens`, `output_tokens` | API가 보고한 사용량 |
| `api_latency_s`, `retrieval_s`, `e2e_s` | 측정 범위를 나눠 기록한 시간 |
| `status`, `error_type` | 처리가 정상적으로 끝났는지, 어떤 예외로 멈췄는지 |

공개 JSONL의 `raw_response`는 상태, 모델과 usage로 축약했습니다. 요청 문맥, 생성 답변과 평가값은 남겨 두었으며 서비스의 응답 ID 등은 공개 기록에서 제거했습니다.

집계는 `experiment.py`의 `summarize(rows, expected)`가 담당합니다. `scoring.py`에도 같은 이름의 함수가 있지만 이 비교 실험에서는 사용하지 않습니다. 실행기는 `scoring.py`에서 `Answer`와 `check_fields`만 가져옵니다.

`expected`는 계획한 총 요청 수입니다. 두 방식의 요청 수가 같다는 전제 아래 각 방식의 분모를 `expected // 2`로 정합니다. 본 실험은 24/2이므로 12입니다. 도중에 멈춰도 분모를 완료 건수로 줄이지 않고, `attempted`, `success`, `unattempted`와 `usage_coverage`를 함께 기록합니다. 미실행을 포함한 일치율만으로 모델을 비교하지 않도록 실행 완료 여부도 함께 읽어야 합니다.

마지막으로 `id`와 `repeat`가 같은 full/rag를 짝지어 정오와 입력 토큰 차이를 기록합니다. 두 방식의 평균뿐 아니라 어떤 질문에서 차이가 났는지 추적하기 위해서입니다. 사람 검토 항목은 `human_review.csv`, 방식별 집계표는 `comparison.csv`에 저장합니다.


<a id="evaluation"></a>
## 정답과 근거를 따로 평가하는 이유

답이 맞았다는 사실과 올바른 근거를 제시했다는 사실은 다릅니다. 같은 수치를 우연히 맞혔거나, 맞는 결론을 엉뚱한 규정으로 설명했을 수도 있습니다. 다음 세 가지를 나눠 확인합니다.

**첫째, 구조화 정답 일치율입니다.** 기준 정답과 `verdict`를 비교하고 수치 질문에서는 `value`도 비교합니다. 153,000원을 묻는 질문에 180,000원을 반환하거나, 답을 구할 수 있는데 `unknown`이라고 답하면 오답으로 처리합니다. 생성에 실패한 문항을 분모에서 빼지 않도록 계획한 문항 수를 기준으로 계산합니다.

**둘째, 근거 ID의 일치입니다.** 모델이 인용한 문단 집합과 기준 근거 집합을 비교합니다. 인용한 문단 중 기준 근거에 해당하는 비율이 precision이고, 기준 근거 중 인용에 포함된 비율이 recall입니다. F1은 두 값을 조화평균한 수치입니다. 문항별로 계산한 뒤 평균을 냅니다.

**셋째, 제공한 문맥과의 관계입니다.** 모델이 인용한 문단이 실제 입력에 있었는지 확인합니다. RAG가 기준 근거를 검색했는지도 따로 계산합니다. 답변이 틀렸을 때 검색 단계에서 이미 정보가 빠졌는지 판단하는 데 필요합니다.

근거 ID F1을 그대로 ‘설명의 신뢰도’로 읽으면 안 됩니다. 문서에는 같은 규정을 다른 표현으로 설명한 문단이 있을 수 있고, 기준 근거 목록에도 꼭 필요하지 않은 문단이 포함될 수 있습니다.

실제로 G03의 출석 조건을 묻는 `dev-06`에서 두 방식은 B02와 C09를 인용했습니다. 기준 목록은 B02, B03, B07입니다. 두 집합에 공통으로 들어 있는 문단은 B02 하나이므로 precision은 1/2, recall은 1/3, F1은 0.4입니다. 그러나 답변은 “G03은 세 회차 모두 출석해야 하므로 두 회차 출석으로는 수료할 수 없다”로 정답과 일치했습니다.

따라서 이 실험은 ID 일치 점수와 의미적 검토를 분리합니다. `human_review.csv`에는 설명이 맞는지, 인용이 주장을 뒷받침하는지 사람이 확인할 칸을 두었습니다. 현재 그 칸은 미검토 상태이며, 자동 점수를 사람 평가처럼 제시하지 않습니다.

### 코드에서 정답과 근거를 계산하는 방법

`check_fields()`는 `scoring.py`, `evidence()`는 `experiment.py`에 있습니다. 둘 다 검증을 통과한 `Answer` 객체와 로컬 기준 정답을 받아 점수를 반환합니다. 모델을 다시 호출하는 평가는 아닙니다.

```python
def check_fields(answer, case, source_ids):
    """판단 종류와 수치 정답을 비교하고 인용 ID의 존재 여부를 검사합니다.

    source_ids는 원문 전체의 문단 ID 집합입니다. 인용 ID가 실제 문서에
    존재하는지 확인할 뿐, RAG 입력에 그 문단이 포함됐는지는 검사하지 않습니다.
    수치 비교는 오차 허용 없이 수행하므로 금액·분 단위 정답에 적합합니다.
    """
    expected = case["expected"]
    match = answer.verdict == expected["verdict"]
    if expected["verdict"] == "value":
        match = match and answer.value == expected["value"]
    return {
        "structured_match": int(match),
        "source_ids_valid": int(
            bool(answer.sources) and set(answer.sources) <= source_ids
        ),
    }


def evidence(answer, case, provided):
    """인용, 기준 근거와 실제 제공 문맥을 서로 다른 집합으로 비교합니다.

    cited는 모델의 인용, gold는 기준 근거, provided는 입력에 넣은 문단입니다.
    set을 사용해 같은 ID를 반복해서 인용해도 점수가 올라가지 않게 합니다.
    citations_in_context와 retrieval_gold_recall은 비교하는 집합이 다릅니다.
    """
    cited, gold = set(answer.sources), set(case['sources'])
    hit = len(cited & gold)
    precision = hit / len(cited) if cited else 0.0
    recall = hit / len(gold) if gold else None
    return {
        'citation_precision': precision,
        'citation_recall': recall,
        'citation_f1': (
            2 * precision * recall / (precision + recall)
            if recall is not None and precision + recall else 0.0
        ),
        'citations_in_context': int(bool(cited) and cited <= set(provided)),
        'retrieval_gold_recall': (
            len(gold & set(provided)) / len(gold) if gold else None
        ),
    }
```

`source_ids_valid`와 `citations_in_context`를 따로 두는 이유를 예로 들어 보겠습니다. RAG가 받지 않은 A02를 답변에 적었다면 A02는 원문에 존재하므로 앞의 검사는 통과할 수 있습니다. 하지만 실제 제공 문맥에는 없으므로 뒤의 검사는 통과하지 못합니다. 반대로 제공된 A10만 인용했다면 두 검사는 모두 통과할 수 있어도 기준 근거를 충분히 포함했다는 뜻은 아닙니다.

실제 `dev-02`의 RAG는 기준 근거 `{A02, A10, C03}` 중 A10만 검색하고 인용했습니다. 인용 precision은 1, recall은 1/3, F1은 0.5이며, 검색된 기준 근거 비율도 1/3입니다. 인용 ID가 모두 입력에 있었으므로 `citations_in_context`는 1이지만 답변은 `unknown`이어서 `structured_match`는 0입니다. 서로 다른 지표가 같은 답변의 다른 측면을 보여 줍니다.

기준 근거가 없는 질문이라면 검색 recall을 0으로 단정하지 않고 `None`으로 반환합니다. 현재 가온 질문에는 근거 목록이 있지만, 새 데이터에 이 함수를 적용할 때는 이런 값을 집계에서 어떻게 다룰지도 결정해야 합니다. 또한 현재 수치 정답은 정수 금액과 시간 중심이므로 정확한 값의 일치를 사용합니다. 소수점 근사값이나 문자열 답변이 필요한 데이터에는 그대로 적용할 수 없습니다.


<a id="results"></a>
## 12문항의 실제 결과

아래 표는 같은 12문항에 대해 각각 한 번씩 생성한 결과입니다. 비용은 뒤에서 설명하는 기본 캐시 쓰기 단가를 적용한 추정치입니다.

| 지표 | 전체 문맥 | BM25 RAG |
|---|---:|---:|
| 구조화 정답 일치 | 12/12 (100%) | 11/12 (91.7%) |
| 근거 ID F1 평균 | 0.906 | 0.892 |
| 인용이 모두 제공 문맥 안에 있는 답변 | 12/12 | 12/12 |
| 기준 근거의 문맥 포함 비율 평균 | 100% | 86.1% |
| 입력 토큰 합계 | 57,308 | 12,778 |
| 출력 토큰 합계 | 843 | 828 |
| API 응답 시간 중앙값 | 6.51초 | 6.31초 |
| API 응답 시간 p95 | 7.08초 | 7.20초 |
| 본 실험 비용 추정 | $0.00648723 | $0.00168558 |

![12문항 비교: 전체 문맥은 12개, RAG는 11개 정답. 입력 토큰은 57,308 대 12,778이며 비용 추정은 0.00648723달러 대 0.00168558달러입니다.](/qa-comparison/comparison.svg)

RAG는 입력 토큰을 **77.7%**, 이번 단가와 캐시 조건에서 비용을 **74.0%** 줄였습니다. 출력 토큰은 843개와 828개로 차이가 작았습니다. 출력 형식과 질문이 같고 입력 문맥만 줄였으므로, 토큰 절감은 주로 입력에서 발생했습니다.

전체 문맥은 12개를 모두 맞혔고 RAG는 한 문항에서 정답을 내지 못했습니다. 다만 문서 한 개와 질문 12개로 얻은 결과입니다. 100%라는 숫자도 이 작은 질문 집합에서의 관찰이지, 문서 QA 전반의 정확도를 뜻하지 않습니다. 작은 표본에서 한 문항 차이는 8.3%포인트가 되므로 실패 사례를 직접 읽는 것이 중요합니다.

<a id="failure"></a>
## 답변이 달라진 할인 계산 문제

두 방식의 답이 달라진 질문은 다음과 같습니다.

> 과거 정규 과정 수료자가 10월 28일 G02 결제를 완료하면 최종 수강료는 몇 원인가요?

이 문제를 풀려면 최소한 G02의 정가와 해당 수강생에게 적용할 할인율을 알아야 합니다. 조기 결제와 재수강 할인이 동시에 적용될 수 있는지도 확인해야 합니다.

전체 문맥 방식은 가격표와 할인 규정을 함께 받았습니다. G02의 정가 180,000원에 재수강 할인 15%를 적용하고, 조기 결제 할인과 중복하지 않는다는 조건에 따라 **153,000원**이라고 답했습니다. A02, A10, C03을 근거로 제시했습니다.

FAQ인 C03에는 153,000원이라는 계산 예시도 직접 적혀 있습니다. 따라서 이 문항의 정답만으로 모델이 여러 문단을 독립적으로 결합해 계산했다고 입증할 수는 없습니다. 여기서는 정답에 필요한 정보가 문맥에 포함됐는지와 최종 답변이 맞는지를 확인합니다.

RAG가 가져온 문단은 A01, A05, A09, A10, C04였습니다. 할인 규정 A10은 있었지만 정가를 담은 A02가 빠졌습니다. 모델은 다음과 같이 답했습니다.

> 정가가 제시되지 않아 최종 수강료를 계산할 수 없습니다. 과거 정규 과정 수료자는 재수강 할인 15%를 적용받고, 조기 결제 할인과는 중복 적용되지 않으므로 G02 정가가 추가로 필요합니다.

RAG의 구조화 결과는 `verdict: unknown`, `value: null`입니다. 전체 문서에는 정답을 계산할 정보가 있으므로 문서 QA의 기준에서는 오답입니다. 동시에 모델이 실제로 받은 다섯 문단만 놓고 보면, 빠진 가격을 추측하지 않고 보류한 이유를 이해할 수 있습니다.

이 사례에서는 검색과 생성을 분리해 볼 수 있습니다. 검색기는 질문에 관련된 할인 문단을 찾았지만, 할인액을 계산하는 데 필요한 가격표까지 가져오지 못했습니다. 생성 모델은 그 부족함을 설명했습니다. 최종 답변만 읽으면 단순히 “RAG가 계산을 못 했다”고 판단할 수 있지만, 입력 문맥을 함께 보면 개선할 위치가 달라집니다.

다음 실험에서는 검색 문단 수를 늘리거나, 과정명과 가격을 확인하는 검색을 추가하는 방법을 비교할 수 있습니다. 다만 이 한 문항의 정답 근거를 강제로 넣어 맞히게 하면 일반적인 검색 성능을 평가하기 어렵습니다. 개발 질문에서 변경을 검토한 뒤 별도로 보관한 질문에서도 같은 개선이 나타나는지 확인해야 합니다.

<a id="latency"></a>
## 토큰을 줄인 만큼 빨라졌을까?

입력 토큰은 크게 줄었지만 응답 시간 중앙값은 6.51초에서 6.31초로 약 0.20초 줄었습니다. RAG의 p95는 오히려 조금 더 길었습니다. 이번처럼 표본이 12개일 때 구현에서 사용하는 nearest-rank 방식의 p95는 사실상 가장 느린 응답이므로, 꼬리 지연을 안정적으로 추정한 값은 아닙니다.

여기서 API 응답 시간은 HTTP 요청을 시작한 뒤 전체 응답을 받을 때까지의 시간입니다. 첫 토큰이 나타나는 시간인 TTFT는 측정하지 않았습니다. 응답 시간에는 네트워크, 서버 처리와 출력 생성이 함께 영향을 줍니다. 입력 토큰의 감소율을 전체 응답 시간의 감소율로 그대로 바꿀 수 없는 이유입니다.

RAG의 질문별 로컬 검색 시간 중앙값은 약 0.000296초, 즉 0.296밀리초였습니다. 31개 문단을 검색하는 작은 실험이므로 여기서는 검색 자체의 시간이 작았습니다. 대규모 인덱스나 외부 검색 서버를 사용하는 환경의 지연으로 일반화할 수는 없습니다.

캐시 조건도 남아 있습니다. 연결 확인에서 사용한 앞의 두 질문을 본 실험에서도 다시 보냈기 때문에 양쪽 방식 모두 캐시 읽기가 발생했습니다. 모든 요청을 처음 보는 상태에서 측정한 결과는 아닙니다. 속도를 더 자세히 비교하려면 캐시 조건과 반복 횟수를 실험 요인으로 따로 정해야 합니다.

모델 이름은 `gpt-6-luna`를 사용했으며 별도의 날짜 고정 스냅샷을 지정하지 않았습니다. 같은 코드와 문서를 사용해도 이후 모델 변경이나 생성 변동으로 답변과 시간이 달라질 수 있습니다. 공개 응답 기록은 이번 실행 결과를 분석하기 위한 기준 자료입니다.

<a id="cost"></a>
## 캐시를 반영한 API 비용

입력 토큰 수에 일반 입력 단가를 곱하는 것만으로는 이번 응답의 비용을 모두 설명할 수 없습니다. API의 `usage.input_tokens_details`에는 캐시 읽기 토큰과 캐시 쓰기 토큰이 함께 기록됐습니다.

2026년 11월 3일 확인한 [GPT-6 Luna의 Standard 단가](https://developers.openai.com/api/docs/models/gpt-6-luna)는 100만 토큰당 일반 입력 0.10달러, 캐시 읽기 0.01달러, 출력 0.50달러입니다. [캐시 문서](https://developers.openai.com/api/docs/guides/prompt-caching)에 따르면 기본 30분 캐시 쓰기는 일반 입력의 1.25배 요율을 사용합니다.

비용을 계산할 때는 입력을 서로 겹치지 않는 세 종류로 나눕니다.

```text
일반 입력 = 전체 입력 − 캐시 읽기 − 캐시 쓰기

비용 = (일반 입력 × 0.10
      + 캐시 읽기 × 0.01
      + 30분 캐시 쓰기 × 0.125
      + 출력 × 0.50) / 1,000,000
```

캐시 쓰기 토큰을 일반 입력에 그대로 둔 채 쓰기 요율을 또 더하면 중복 계산합니다. 반대로 쓰기 요율을 무시하면 비용을 낮게 추정하게 됩니다. 공개 코드의 `analysis/cache_costs.py`는 원본 사용량에서 각 항목을 분리해 계산합니다.

| 구분 | 호출 수 | 기본 30분 쓰기 가정 |
|---|---:|---:|
| 연결 확인 | 4 | $0.00160195 |
| 본 실험 | 24 | $0.00817281 |
| 합계 | 28 | **$0.00977476** |

이번 요청은 캐시 쓰기 TTL을 지정하지 않았고, 응답에는 TTL별 쓰기 토큰의 분할이 없었습니다. 따라서 기본 30분을 가정했습니다. 12시간 쓰기의 높은 요율을 적용하면 합계는 약 0.01817달러가 됩니다. 이 값도 함께 기록해 가정에 따른 차이를 확인할 수 있게 했습니다. 응답의 최대 보관 설정만 보고 쓰기 TTL을 확정하지는 않았습니다.

위 금액은 API 사용량과 공개 단가로 계산한 **토큰 비용 추정치**입니다. 청구서나 크레딧 차감액을 확인한 수치는 아닙니다. 최초 실행기의 일반 입력·캐시 읽기 계산값은 원본 요약에 남겨 두었으며, 이 글의 비용 표에는 캐시 쓰기를 반영한 후처리 값을 사용했습니다.

### `priced()`: 사용량에서 비용을 다시 계산하기

실행 중의 `usage_cost()`는 입력·캐시 읽기·출력·추론 토큰을 기록합니다. 이 함수의 `token_cost_estimate_usd`는 캐시 쓰기 할증을 반영하기 전 계산이므로, 최종 비용에는 후처리 함수 `priced()`의 결과를 사용합니다. 원래 사용량을 남겨 두면 응답을 다시 생성하지 않고도 비용 계산을 검토할 수 있습니다.

아래는 `analysis/cache_costs.py`의 `priced()` 전체입니다. 같은 파일에서 `D`는 `decimal.Decimal`, `MILLION`은 `D(1000000)`으로 정의합니다.

```python
def priced(row, config):
    """저장된 usage를 요금 종류별로 나눠 한 요청의 비용을 계산합니다.

    입력 합계에서 캐시 읽기와 쓰기를 빼 일반 입력을 구합니다.
    TTL별 쓰기 내역이 있으면 각각의 단가를 적용합니다. 내역이 없으면
    기본 30분 가정과 12시간 요율을 적용한 값을 둘 다 반환합니다.
    결과는 Decimal로 계산하고 JSON 저장을 위해 문자열로 변환합니다.
    """
    usage = row['raw_response']['usage']
    details = usage['input_tokens_details']
    total, cached = usage['input_tokens'], details['cached_tokens']
    written = details['cache_write_tokens']
    regular = total - cached - written
    assert min(regular, cached, written, usage['output_tokens']) >= 0
    assert total == row['input_tokens'] and cached == row['cached_input_tokens']

    rate = D(str(config['input_usd_per_million']))
    base = regular * rate + cached * D(str(config['cached_usd_per_million']))
    base += usage['output_tokens'] * D(str(config['output_usd_per_million']))
    split = details.get('cache_write_tokens_details')
    if split is not None:
        w30, w12 = split['ttl_30m_tokens'], split['ttl_12h_tokens']
        assert w30 + w12 == written and min(w30, w12) >= 0
        low = high = (
            base + rate * (w30 * D('1.25') + w12 * D('2.5'))
        ) / MILLION
        assumption = 'usage_ttl_breakdown'
    else:
        assert 'prompt_cache_options' not in row['request']
        low = (base + written * rate * D('1.25')) / MILLION
        high = (base + written * rate * D('2.5')) / MILLION
        assumption = 'default_30m_assumed_no_ttl_breakdown'

    return {
        'id': row['id'],
        'arm': row['arm'],
        'repeat': row['repeat'],
        'ordinary_input_tokens': regular,
        'cached_input_tokens': cached,
        'cache_write_tokens': written,
        'input_tokens': total,
        'output_tokens': usage['output_tokens'],
        'assumption': assumption,
        'cache_aware_estimate_usd': str(low),
        'write_12h_sensitivity_usd': str(high),
    }
```

처음 두 `assert`는 입력 내역에 음수가 생기지 않았는지와 다른 필드에 저장한 사용량이 일치하는지 확인합니다. `Decimal`에는 `float`를 바로 전달하지 않고 문자열을 거쳐 전달해 요금의 소수를 십진수로 계산합니다.

`base`는 일반 입력, 캐시 읽기와 출력의 비용을 합한 값입니다. 쓰기 비용만 나중에 더하므로 같은 입력 토큰을 중복해서 계산하지 않습니다. 추론 토큰은 API의 출력 토큰에 포함되므로 출력 비용에 다시 더하지 않습니다.

TTL별 내역이 있으면 30분과 12시간 쓰기 토큰 수를 확인해 각각 계산합니다. 이때 `low`와 `high`는 같은 값입니다. 내역이 없으면 요청에서 TTL을 지정하지 않았는지 확인하고 두 가정을 적용합니다. `low`와 `high`는 통계적 신뢰 구간이 아니라 요금 가정을 달리한 계산 결과입니다.

`aggregate()`는 이 결과를 방식별, 실행별로 더합니다. 현재 `main()`은 연결 확인 4회와 본 실험 24회의 저장된 결과를 대상으로 건수와 원장을 검사합니다. 요청 수가 다른 새 실험을 분석할 때는 그 실험에 맞춰 집계 대상과 검증 조건을 바꿔야 합니다.


<a id="practice"></a>
## 직접 실행하고 확인하기

코드, 문서, 질문과 공개용 응답 기록은 [실험 폴더](https://github.com/teimakes/teimakes.github.io/tree/main/public/qa-comparison)에 있습니다. 모델을 새로 호출하지 않아도 검색 누락과 채점 결과를 확인할 수 있습니다.

```bash
git clone https://github.com/teimakes/teimakes.github.io.git
cd teimakes.github.io/public/qa-comparison
python analysis/inspect_example.py
python analysis/cache_costs.py
```

첫 번째 Python 명령은 원문에 BM25를 다시 적용하고 `dev-02`에서 검색된 문단과 빠진 기준 근거를 출력합니다. 이어서 저장된 전체 문맥 답변과 RAG 답변을 읽습니다. 두 번째 명령은 저장된 usage로 비용을 다시 계산합니다. 두 명령 모두 API 키가 필요 없고 생성 요청을 보내지 않습니다.

검색 결과는 다음과 같습니다.

```text
검색 문단: A01, A05, A09, A10, C04
검색에서 빠진 기준 근거: A02, C03
```

코드를 더 살펴볼 때는 아래 순서가 도움이 됩니다.

| 파일 | 확인할 내용 |
|---|---|
| `data/handbook.txt` | 정가와 할인 조건이 어느 문단에 있는가 |
| `data/dev.jsonl` | 질문별 정답과 기준 근거가 무엇인가 |
| `retrieval.py` | 어떤 문자열 단위로 문단 점수를 계산하는가 |
| `prompts.py` | 답변 보류와 근거 인용을 어떻게 지시하는가 |
| `scoring.py` | 정답으로 인정하는 조건이 무엇인가 |
| `outputs/dev/results.jsonl` | 실제 입력 문맥, 모델 답변과 usage가 무엇인가 |
| `analysis/cache_costs.py` | 캐시 토큰을 비용에 어떻게 반영하는가 |

실행기의 오프라인 검사는 다음 명령으로 수행합니다. 원래 실험 코드에 대한 검사 20개는 통과했습니다.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python experiment.py plan --limit 2 --out plans/new-smoke.json
```

마지막 명령은 새 연결 확인 계획을 만들 뿐 API를 호출하지 않습니다. 실제 유료 실행에는 본인 계정의 모델 접근 권한, 단가와 예산 확인이 필요합니다. 공개된 설정에는 원래 실험의 실행 차단일과 호출 이력이 남아 있으므로 그대로 재호출하는 대신 [README의 새 실행 절차](https://github.com/teimakes/teimakes.github.io/blob/main/public/qa-comparison/README.md)를 읽고 조건을 확인해야 합니다.

<a id="next"></a>
## MuSiQue와 파인튜닝으로 확장하기

짧은 가상 안내서는 실패 원인을 확인하기 쉽지만 문서 QA의 다양한 상황을 대표하지는 않습니다. 다음 단계에서는 여러 문단의 정보를 결합해야 하는 공개 벤치마크 [MuSiQue](https://github.com/StonyBrookNLP/musique)로 범위를 넓힐 수 있습니다.

이번 코드에는 MuSiQue의 질문·후보 문단을 생성 입력으로, 정답·근거 정보를 평가 대상으로 분리하는 어댑터를 두었습니다. 질문을 분해한 정보나 정답 근거 표시가 모델 입력에 섞이지 않도록 하는 것이 목적입니다. 현재 생성 실행기는 가온 문서용이므로 MuSiQue의 문자열 답변 형식과 공식 평가기를 연결하는 작업은 남아 있습니다. 이 글에서는 MuSiQue를 실행한 결과를 제시하지 않습니다.

확장할 때도 비교 범위를 먼저 정해야 합니다. 예제에 주어진 후보 문단 안에서 검색하는 경우와 전체 Wikipedia에서 검색하는 경우는 다른 문제입니다. 첫 비교에서는 동일한 후보 문단을 두 방식에 제공하고, 그 안에서 검색이 토큰과 정답에 어떤 영향을 주는지 보는 편이 현재 실험과 연결됩니다.

파인튜닝에는 별도의 학습 질문과 답변, 검증 자료, 지원 모델과 학습 비용 검토가 필요합니다. 평가용 24문항을 그대로 학습시키면 문서 QA 능력과 평가 문항의 암기를 구분하기 어렵습니다. 외부 문서를 계속 제공하면서 답변 형식을 학습할지, 문서의 사실을 모델에 학습시킬지도 먼저 결정해야 합니다. 두 목적은 평가 방법이 다릅니다.

이번 비교에서 확인한 것은 작은 문서에서도 검색이 입력을 크게 줄일 수 있다는 점, 그리고 질문과 관련된 문단을 찾는 것과 답을 완성할 근거를 모두 찾는 것이 같지 않다는 점입니다. 다음 실험에서는 누락된 근거를 더 잘 가져오는 변경이 다른 질문에서도 도움이 되는지, 그 대가로 입력과 시간이 얼마나 늘어나는지를 함께 측정할 수 있습니다.

## 코드와 참고 자료

- [실험 코드·문서·질문·응답 기록](https://github.com/teimakes/teimakes.github.io/tree/main/public/qa-comparison)
- [문항별 캐시 비용 계산 결과](/qa-comparison/outputs/cache_costs.json)
- [GPT-6 Luna 모델 및 가격](https://developers.openai.com/api/docs/models/gpt-6-luna)
- [OpenAI Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
- [MuSiQue 공식 저장소와 평가 코드](https://github.com/StonyBrookNLP/musique)
