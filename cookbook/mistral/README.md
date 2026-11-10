# Mistral API Cookbook (한국어, 실행용 스크립트)

Mistral 공식 SDK(`mistralai`)로 바로 실행해 볼 수 있는 레시피 19개 모음입니다.
공식 [Mistral Cookbook](https://github.com/mistralai/cookbook)의 주제를 한국어 실전 스크립트 형태로 재구성했습니다.

## 준비

```bash
export MISTRAL_API_KEY="여기에-본인-키"   # https://console.mistral.ai/ 에서 발급
pip install -r requirements.txt
```

API 키는 코드에 절대 하드코딩하지 말고 환경변수로만 관리하세요.

## 레시피 목록

| 번호 | 파일 | 주제 |
| --- | --- | --- |
| 01 | `01_chat.py` | Chat Completion 기본, 시스템 프롬프트, 모델 비교 |
| 02 | `02_stream.py` | 스트리밍 토큰 출력 |
| 03 | `03_json_schema.py` | JSON Schema 구조화 출력 |
| 04 | `04_function_calling.py` | 도구 호출 멀티턴 루프 |
| 05 | `05_embeddings_classify.py` | 임베딩 few-shot 텍스트 분류 |
| 06 | `06_embeddings_clustering.py` | 임베딩 + 직접 구현한 K-means 클러스터링 |
| 07 | `07_rag_basic.py` | RAG from scratch (분할·검색·생성) |
| 08 | `08_rag_router.py` | Function calling으로 다중 소스 RAG 라우팅 |
| 09 | `09_ocr_to_markdown.py` | PDF OCR → Markdown 저장 |
| 10 | `10_ocr_structured.py` | OCR + JSON Schema 문서 구조화 추출 |
| 11 | `11_vision_pixtral.py` | Pixtral 이미지 이해 + 구조화 출력 |
| 12 | `12_moderation.py` | Moderation API 유해 콘텐츠 분류 |
| 13 | `13_sampling_params.py` | temperature·top_p·top_k 샘플링 비교 |
| 14 | `14_llm_judge.py` | LLM-as-a-Judge 답변 평가 |
| 15 | `15_synthetic_data.py` | 합성 데이터 JSONL 생성 (+ fine-tune 작업 생성) |
| 16 | `16_batch_inference.py` | 배치 추론 잡 생성·조회 |
| 17 | `17_text2sql.py` | 자연어 → SQL (스키마 컨텍스트) |
| 18 | `18_agents_api.py` | Agents API 에이전트 생성·실행·삭제 |
| 19 | `19_vibe_recipes.md` | 코딩 에이전트 Vibe 활용 레시피 |

## 실행 예시

```bash
python 01_chat.py
python 07_rag_basic.py
python 09_ocr_to_markdown.py paper.pdf
python 11_vision_pixtral.py receipt.jpg
```

파일 인자가 필요한 스크립트(09, 10, 11)는 인자를 빼고 실행하면 사용법을 출력합니다.

## 요금 안내

실제 API 사용량에 따른 요금은 [La Plateforme](https://console.mistral.ai/)의 과금 정책과
[가격표](https://mistral.ai/pricing)를 따릅니다. fine-tuning/배치/OCR은 별도 과금 항목일 수 있으니
실행 전에 가격표를 확인하세요.

## 참고 자료

- [Mistral AI 공식 문서](https://docs.mistral.ai/)
- [공식 Cookbook (GitHub)](https://github.com/mistralai/cookbook)
- [mistralai Python SDK](https://github.com/mistralai/client-python)
