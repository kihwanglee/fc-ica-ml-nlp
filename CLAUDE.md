# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

"LLM 이전의 AI를 만나다" 특강 자료 및 예제 프로젝트. 고전적 머신러닝(분류, 군집화)과 한국어 자연어 처리(형태소 분석, LDA 토픽 모델링)를 다루며, Flask 웹 앱 실습까지 포함한다.

## 환경 설정

의존성 관리는 `uv`를 사용한다. Python 버전은 3.12.

```bash
uv sync                          # 의존성 설치
uv add <패키지명>                 # 패키지 추가
uv run python script/<파일>.py   # 스크립트 실행
```

## 주요 라이브러리

- `kiwipiepy`: 한국어 형태소 분석 (명사 추출, `NN` 태그 필터링)
- `gensim`: LDA 토픽 모델 학습
- `pyLDAvis`: LDA 결과 인터랙티브 HTML 시각화
- `scikit-learn`: RandomForest 분류 모델 (MNIST)
- `matplotlib`: 시각화 — 한글 폰트는 플랫폼별로 분기 설정 (아래 참조)
- `flask`: 로컬 웹 서버
- `pillow`: 캔버스 이미지 전처리 (base64 PNG → 28×28 배열)
- `joblib`: 학습된 모델 직렬화 (`sklearn` 내장)

## 스크립트 구조

### `script/topic_model.py`
한국어 뉴스 헤드라인 50건으로 LDA 토픽 모델링.
- 파이프라인: `Kiwi.analyze()` → 명사 추출 → `corpora.Dictionary` + `doc2bow` → `LdaModel` → `pyLDAvis.save_html()`
- 출력: `lda_result.html` (시각화), `lda_analysis.md` (토픽별 키워드·문서 배정 리포트)
- 파라미터: `num_topics=6`, `passes=50`

### `script/recog_digits.py`
MNIST 손글씨 숫자 분류 데모.
- `fetch_openml('mnist_784')` → 60,000건 학습 / 10,000건 테스트
- `RandomForestClassifier(n_estimators=100)` → 약 97% 정확도
- matplotlib으로 예측 결과 2×5 그리드 시각화

### `script/app.py`
MNIST 모델을 Flask로 서빙하는 로컬 웹 앱.
- 실행: `uv run python script/app.py` → `http://127.0.0.1:8080`
- 최초 실행 시 모델 학습 후 `model/mnist_rf.pkl`로 저장, 이후 재사용
- `/predict` 엔드포인트: base64 PNG 수신 → Pillow로 28×28 리사이즈 → 추론 → 확률 반환
- macOS 포트 5000 충돌 회피를 위해 8080 사용 (5000은 AirPlay Receiver가 점유)

## 한글 폰트 설정 패턴

matplotlib 사용 스크립트(`topic_model.py`, `recog_digits.py`)는 공통으로 플랫폼별 한글 폰트를 설정한다.

```python
import platform, matplotlib.pyplot as plt, matplotlib.font_manager as fm
_system = platform.system()
if _system == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif _system == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    _nanum = [f.name for f in fm.fontManager.ttflist if 'Nanum' in f.name]
    if _nanum:
        plt.rcParams['font.family'] = _nanum[0]
plt.rcParams['axes.unicode_minus'] = False
```

## 산출물 및 gitignore

- `lda_result.html` / `lda_analysis.md`: `topic_model.py` 실행 산출물 (루트에 생성)
- `model/`: 학습된 모델 파일 — `.gitignore`에 등록되어 있음 (용량 큼)
