import numpy as np
import matplotlib.pyplot as plt

#set random vector
np.random.seed(42)
X = np.random.randn(100)

#--------------
# pre processing
#----------------

def minmax_normalize(X):
  X_min = X.min()
  X_max = X.max()
  return (X - X_min) / (X_max - X_min + 1e-8)

def z_score_normalize(X):
  mx = X.mean()
  sigma = X.std()
  return (X - mx) / (sigma + 1e-8)

def simple_whitening(X):
  Xc = X - X.mean()
  return Xc / (np.sqrt(np.var(Xc)) + 1e-8)

x_minmax = minmax_normalize(X)
x_zscore = z_score_normalize(X)
x_white = simple_whitening(X)

# visualization
plt.figure()
plt.plot(x_minmax, label = "minmax")
plt.plot(x_zscore, label = "zscore")
plt.plot(x_white = "whitening")
plt.legend()
plt.show

#-----------------
# activation functions
#-------------------

def identity(a): return a
def step(a): return np.where(a>0,1,0)
def sigmoid(a): return 1/(1+np.exp(-a))
def relu(a): return np.maximum(0,a)

Y_id = identity(X)
Y_step = step(X)
Y_sig = sigmoid(X)
Y_relu = relu(X)

#-----------------
#softmax function
#-------------------
a = np.random.randn(10)
exp_a = np.exp(a-np.max(a))
y_soft = exp_a/np.sum(exp_a)

plt.figure()
plt.bar(np.arange(len(a)),y_soft)
plt.show
