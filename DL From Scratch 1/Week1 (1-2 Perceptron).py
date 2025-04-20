import numpy as np

class perceptron:
  def __init__(self, input_size):
    self.w = np.zeros(input_size)
    self.b =0.0

  def predict(self,x):
    linear_output = np.dot(x,self.w)+self.b
    return 1 if linear_output>0 else 0

  def train(self,X,y, lr = 0.1, epochs =10):
    # lr = learning rate
    for _ in range(epochs):
      for xi, yi in zip(X,y):
        pred = self.predict(xi)
        error = yi - pred
        #weights and bias update
        self.w += lr * error * xi
        self.b += lr * error
        

#And Gate
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

p = perceptron(input_size =2)
p.train(X,y, lr = 0.1, epochs =20)

print("And weights =", p.w)
print("And bias=", p.b)
print("prediction result=", [p.predict(xi) for xi in X])
