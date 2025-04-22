import numpy as np
from sklearn.datasets import load_digits

# 1) 데이터 로드 및 전처리
digits = load_digits()
X = digits.data.astype(np.float32) / 16.0   # 픽셀 0~16 → 0~1
y = digits.target.astype(int)

x_train = X[:1000]
y_train = np.eye(10)[y[:1000]]
x_test  = X[1000:1300]
y_test  = np.eye(10)[y[1000:1300]]

# 2) set the batches
def get_mini_batches(X, y, batch_size=64):
    m = X.shape[0]                       # 전체 샘플 수
    indices = np.random.permutation(m)   # 무작위로 섞은 인덱스
    for i in range(0, m, batch_size):
        idx = indices[i:i+batch_size]    # 한 덩어리(배치) 인덱스
        yield X[idx], y[idx]             # 배치 하나씩 반환

# 3) layer set (파라미터 초기화)
input_dim, hidden_dim, output_dim = 64, 32, 10

w1 = np.random.randn(input_dim, hidden_dim) * 0.01
b1 = np.zeros((1, hidden_dim))

w2 = np.random.randn(hidden_dim, output_dim) * 0.01
b2 = np.zeros((1, output_dim))

# 학습 설정
learning_rate = 0.1
epochs = 5
batch_size = 64

# 활성화/소프트맥스 함수 정의
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 4) 학습 루프
for epoch in range(1, epochs+1):
    epoch_loss = 0.0
    for X_batch, y_batch in get_mini_batches(x_train, y_train, batch_size):
        # 4.1 forward
        z1 = X_batch @ w1 + b1
        a1 = sigmoid(z1)
        z2 = a1 @ w2 + b2
        a2 = softmax(z2)

        # 4.2 loss (교차 엔트로피)
        m_batch = X_batch.shape[0]
        loss = -np.sum(y_batch * np.log(a2 + 1e-8)) / m_batch
        epoch_loss += loss

        # 4.3 backward
        dz2 = (a2 - y_batch) / m_batch
        dw2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ w2.T
        dz1 = da1 * (a1 * (1 - a1))
        dw1 = X_batch.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # 4.4 parameter update
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1

    # 5) 에폭마다 평가
    #   Train accuracy
    z1_train = x_train @ w1 + b1
    a1_train = sigmoid(z1_train)
    z2_train = a1_train @ w2 + b2
    a2_train = softmax(z2_train)
    y_pred_train = np.argmax(a2_train, axis=1)
    y_true_train = np.argmax(y_train, axis=1)
    train_acc = np.mean(y_pred_train == y_true_train)

    #   Test accuracy
    z1_test = x_test @ w1 + b1
    a1_test = sigmoid(z1_test)
    z2_test = a1_test @ w2 + b2
    a2_test = softmax(z2_test)
    y_pred_test = np.argmax(a2_test, axis=1)
    y_true_test = np.argmax(y_test, axis=1)
    test_acc = np.mean(y_pred_test == y_true_test)

    print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}, "
          f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
