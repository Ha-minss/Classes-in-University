import numpy as np
from tensorflow.keras.datasets import mnist

# 1) 데이터 불러와 펼치고 정규화/원-핫
(x, t), _ = mnist.load_data()
X = x.reshape(-1, 784)/255.0
Y = np.eye(10)[t]

#2. 파라미터 초기화
w1 = np.random.randn(784,50)*0.01; b1 = np.zeros(50)
w2 = np.random.randn(50,10)*0.01; b2 = np.zeros(10)
lr,epochs, bs = 0.1, 5, 100

# 학습루프
for _ in range(epochs):
  for i in range(0,X.shape[0],bs):
        xb = X[i:i+bs]; yb = Y[i:i+bs]
        #순전파
        h = 1/(1+np.exp(-(xb@w1 + b1)))
        p = np.exp(h@w2 + b2)
        p /= p.sum(axis =1, keepdims = True)
        #역전파
        e = (p-yb)/bs
        w2 -= lr * (h.T@e); b2 -= lr*e.sum(0)
        dh = (e@w2.T) * h*(1-h)
        w1 -= lr*(xb.T@dh); b1 -= lr*dh.sum(0)
