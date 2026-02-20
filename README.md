# fc-ica-ml-nlp

## 개요

"LLM 이전의 AI를 만나다" 특강 자료와 예제 스크립트 모음.
고전적 머신러닝(분류·군집화)과 한국어 자연어 처리(형태소 분석·토픽 모델링)를 다루며,
학습한 모델을 Flask 웹 앱으로 서빙하는 실습까지 포함한다.

## 환경

- Python 3.12 / 의존성 관리: `uv`

```bash
uv sync          # 의존성 설치
```

## 예제 스크립트

| 스크립트 | 내용 |
|----------|------|
| `script/topic_model.py` | 한국어 뉴스 헤드라인 50건으로 LDA 토픽 모델링. 결과를 `lda_result.html`(인터랙티브 시각화)과 `lda_analysis.md`(분석 리포트)로 저장 |
| `script/recog_digits.py` | MNIST 손글씨 숫자 데이터셋으로 RandomForest 분류 모델 학습 및 예측 결과 시각화 |
| `script/app.py` | MNIST 모델을 Flask로 서빙하는 로컬 웹 앱. 트랙패드/마우스로 숫자를 그리면 실시간 예측 |

### 실행

```bash
# 토픽 모델링 (lda_result.html, lda_analysis.md 생성)
uv run python script/topic_model.py

# MNIST 숫자 인식 (matplotlib 시각화)
uv run python script/recog_digits.py

# 웹 앱 (http://127.0.0.1:8080)
uv run python script/app.py
```

## 강의 자료

`doc/fc-ica-ml-nlp.md` — 회귀·분류·군집화, 형태소 분석, 키워드/토픽 분석, 웹 앱 실습 내용을 포함한 특강 자료 전문
