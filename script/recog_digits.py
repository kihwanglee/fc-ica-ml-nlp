from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 플랫폼별 한글 폰트 설정 (한글 깨짐 방지)
_system = platform.system()
if _system == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif _system == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:  # Linux 등
    # 설치된 나눔 계열 폰트 중 첫 번째를 사용, 없으면 기본값 유지
    _nanum = [f.name for f in fm.fontManager.ttflist if 'Nanum' in f.name]
    if _nanum:
        plt.rcParams['font.family'] = _nanum[0]
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 1. 데이터 로드
print("데이터 로딩 중...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target

# 2. 학습/테스트 분리 (60,000 학습 / 10,000 테스트)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# 3. 픽셀값 정규화 (0~255 → 0~1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# 4. 모델 학습
print("모델 학습 중...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. 예측 및 정확도 측정
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"정확도: {accuracy:.4f}")  # 약 97% 달성!

# 6. 시각화: 예측 결과 확인
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"예측: {y_pred[i]}\n실제: {y_test[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()