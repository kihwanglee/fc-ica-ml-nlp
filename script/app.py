"""
MNIST 숫자 인식 웹 앱
실행: uv run python script/app.py
접속: http://localhost:5000
"""
import os
import base64
import io
import joblib
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'mnist_rf.pkl')

app = Flask(__name__)
model = None


# ── 모델 학습 및 저장 ────────────────────────────────────────────────────────

def train_and_save():
    print("[1/3] MNIST 데이터 로딩 중...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    X_train, X_test = X_train / 255.0, X_test / 255.0

    print("[2/3] 모델 학습 중... (수 분 소요될 수 있습니다)")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"      정확도: {acc:.4f}")

    print("[3/3] 모델 저장 중...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"      저장 완료 → {MODEL_PATH}")
    return clf


def load_or_train():
    global model
    if os.path.exists(MODEL_PATH):
        print(f"저장된 모델 로드 중... ({MODEL_PATH})")
        model = joblib.load(MODEL_PATH)
        print("모델 로드 완료")
    else:
        print("저장된 모델 없음 → 새로 학습합니다.")
        model = train_and_save()


# ── HTML 템플릿 ───────────────────────────────────────────────────────────────

HTML = """\
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MNIST 숫자 인식</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Apple SD Gothic Neo', 'Malgun Gothic', 'NanumGothic', sans-serif;
    background: #1a1a2e;
    color: #eee;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem 1rem;
    min-height: 100vh;
  }
  h1 { font-size: 1.8rem; margin-bottom: 0.3rem; }
  .subtitle { color: #888; margin-bottom: 1.5rem; font-size: 0.9rem; }
  canvas {
    border: 3px solid #444;
    border-radius: 12px;
    background: #000;
    cursor: crosshair;
    touch-action: none;
    display: block;
  }
  .buttons {
    display: flex;
    gap: 0.8rem;
    margin-top: 1rem;
  }
  button {
    padding: 0.6rem 2rem;
    font-size: 1rem;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-weight: bold;
    transition: opacity 0.15s;
  }
  button:hover { opacity: 0.85; }
  #predictBtn { background: #4ade80; color: #111; }
  #clearBtn   { background: #f87171; color: #fff; }
  #result {
    margin-top: 1.5rem;
    font-size: 2.8rem;
    font-weight: bold;
    min-height: 3.5rem;
    letter-spacing: 0.05em;
  }
  #probWrap { margin-top: 0.8rem; width: 300px; }
  .prob-row {
    display: flex;
    align-items: center;
    margin: 4px 0;
    font-size: 0.9rem;
  }
  .digit-label { width: 20px; text-align: right; margin-right: 8px; color: #aaa; }
  .bar-bg {
    flex: 1;
    background: #2e2e4e;
    border-radius: 4px;
    height: 16px;
    overflow: hidden;
  }
  .bar-fill {
    height: 16px;
    border-radius: 4px;
    background: #4ade80;
    transition: width 0.35s ease;
  }
  .bar-fill.best { background: #facc15; }
  .pct { width: 50px; text-align: right; color: #888; font-size: 0.8rem; }
  #hint {
    margin-top: 2rem;
    font-size: 0.8rem;
    color: #555;
  }
</style>
</head>
<body>
<h1>✏️ MNIST 숫자 인식</h1>
<p class="subtitle">트랙패드나 마우스로 숫자(0~9)를 그린 뒤 예측 버튼을 누르세요.</p>

<canvas id="canvas" width="280" height="280"></canvas>

<div class="buttons">
  <button id="predictBtn">예측</button>
  <button id="clearBtn">지우기</button>
</div>

<div id="result"></div>
<div id="probWrap"></div>
<p id="hint">캔버스 중앙에 크게 그릴수록 정확도가 높아집니다.</p>

<script>
const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');

// 검정 배경 초기화
ctx.fillStyle = '#000';
ctx.fillRect(0, 0, 280, 280);

// 흰색 굵은 선으로 그리기 (MNIST 형식: 흑배경 + 백선)
ctx.strokeStyle = '#fff';
ctx.lineWidth   = 20;
ctx.lineCap     = 'round';
ctx.lineJoin    = 'round';

let drawing = false;

function getPos(e) {
  const r   = canvas.getBoundingClientRect();
  const src = e.touches ? e.touches[0] : e;
  return { x: src.clientX - r.left, y: src.clientY - r.top };
}

// 마우스 이벤트
canvas.addEventListener('mousedown', e => {
  drawing = true;
  const p = getPos(e);
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
});
canvas.addEventListener('mousemove', e => {
  if (!drawing) return;
  const p = getPos(e);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
});
canvas.addEventListener('mouseup',    () => { drawing = false; });
canvas.addEventListener('mouseleave', () => { drawing = false; });

// 터치 이벤트 (트랙패드 핀치/스크롤 방지)
canvas.addEventListener('touchstart', e => {
  e.preventDefault();
  drawing = true;
  const p = getPos(e);
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
}, { passive: false });
canvas.addEventListener('touchmove', e => {
  e.preventDefault();
  if (!drawing) return;
  const p = getPos(e);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
}, { passive: false });
canvas.addEventListener('touchend', () => { drawing = false; });

// 지우기
document.getElementById('clearBtn').addEventListener('click', () => {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, 280, 280);
  document.getElementById('result').textContent   = '';
  document.getElementById('probWrap').innerHTML   = '';
});

// 예측
document.getElementById('predictBtn').addEventListener('click', async () => {
  const resultEl  = document.getElementById('result');
  const probWrap  = document.getElementById('probWrap');

  resultEl.textContent = '예측 중...';
  probWrap.innerHTML   = '';

  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: canvas.toDataURL('image/png') }),
    });
    const data = await res.json();

    resultEl.textContent = `예측 결과: ${data.prediction}`;

    const maxProb = Math.max(...data.probabilities.map(p => p.prob));
    probWrap.innerHTML = data.probabilities.map(({ digit, prob }) => {
      const pct    = (prob * 100).toFixed(1);
      const isBest = prob === maxProb;
      return `<div class="prob-row">
        <span class="digit-label">${digit}</span>
        <div class="bar-bg">
          <div class="bar-fill ${isBest ? 'best' : ''}" style="width:${pct}%"></div>
        </div>
        <span class="pct">${pct}%</span>
      </div>`;
    }).join('');
  } catch (err) {
    resultEl.textContent = '오류가 발생했습니다.';
  }
});
</script>
</body>
</html>
"""


# ── Flask 라우트 ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/predict', methods=['POST'])
def predict():
    # 캔버스 이미지(base64 PNG) → 28×28 그레이스케일 배열
    img_b64   = request.json['image']
    img_bytes = base64.b64decode(img_b64.split(',')[1])
    img       = Image.open(io.BytesIO(img_bytes)).convert('L')  # 그레이스케일
    img       = img.resize((28, 28), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.reshape(1, -1)

    pred  = model.predict(arr)[0]
    proba = model.predict_proba(arr)[0]

    return jsonify({
        'prediction': pred,
        'probabilities': [
            {'digit': cls, 'prob': float(p)}
            for cls, p in zip(model.classes_, proba)
        ],
    })


# ── 진입점 ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    load_or_train()
    print("\n서버 시작 → http://127.0.0.1:8080\n")
    app.run(host='0.0.0.0', port=8080, debug=False)
