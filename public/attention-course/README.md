# 어텐션과 트랜스포머: 5주 PyTorch 실습

각 주는 강의 90분 + 실습 90분 + 과제 2~4시간으로 구성합니다. 선형대수, Python 함수·클래스, 미분의 기본 지식을 전제로 합니다.

1. week01.py: 분류 정답만으로 중요한 위치 선택을 학습합니다.
2. week02.py: Q·K·V를 학습해 질문에 맞는 카드 값을 검색합니다.
3. week03.py: 인과적 셀프 어텐션으로 지연 복사를 학습합니다.
4. week04.py: 두 관계를 동시에 검색하고 1헤드/2헤드를 비교합니다.
5. week05.py: 디코더 언어 모델을 학습하고 체크포인트를 복원해 생성합니다.

## 설치와 실행

검증 환경: Python 3.12.14, PyTorch 2.15.0+cpu, CPU 1스레드.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch==2.15.0
python week01.py
python week02.py
python week03.py
python week04.py
python week04.py --heads 1 --out runs/week04-head1
python week05.py
```

Linux/Windows CPU 휠은 pip 명령에 `--index-url https://download.pytorch.org/whl/cpu`를 붙여 설치할 수 있습니다. 기본 설치 명령은 운영체제에 맞는 기본 PyTorch 배포를 사용합니다. 실험은 CPU로 실행되며 CUDA를 요구하지 않습니다.

각 파일은 독립 실행되며 외부 API 키·데이터 다운로드가 없습니다. 결과와 모델 파일은 `--out`에 지정한 경로에 저장됩니다. `--seed`, `--steps`를 바꾸어 실험하세요. 로그와 그래프의 원본은 weekXX-metrics.json과 results.json에 있습니다. 측정 시간에는 라이브러리 설치·import 시간이 포함되지 않습니다.

모든 과제는 학습이 되는지와 정보를 올바르게 사용하는지를 살펴보는 통제된 합성 실험입니다. 자연어 이해 성능을 측정하는 벤치마크가 아닙니다. 5주차에는 시드에 따른 일반화 실패도 포함되어 있으므로 높은 정확도의 실행만 선택하지 말고 전체 기록을 읽으세요.

5주차는 학습·검증·테스트의 값 조합을 분리하며 검증으로 best.pt를 선택한 뒤 테스트합니다. 모델 구성과 어휘가 같은 상태에서 로드해야 합니다. CPU 시드 고정은 환경 간 완전한 비트 일치를 보장하지 않습니다.
