import time
import numpy as np
from PIL import Image
from scipy import ndimage

def image2vector(image):
    return image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)

def normalizeRows(x):
    x = x / np.linalg.norm(x, axis = 1, keepdims = True)
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis = 1, keepdims = True)

def tanh(x):
    return np.tanh(x)

def ReLU(x):
    return np.max(0, x)

def Leaky_ReLU(x):
    return np.max(0.01 * x, x)

def L1(yhat, y):
    return np.sum(np.abs(y - yhat))

def L2(yhat, y):
    return np.dot((y - yhat),(y - yhat).T)

m = 100000
alpha = 0.1
w1 = np.random.randn(4, 2)
w2 = np.random.randn(1, 4)
b1 = np.zeros((4, 1))
b2 = np.zeros((1, 1))
while True:
    x_train = np.random.rand(2, m)
    y_train = np.where(x_train[0] > x_train[1], 0, 1)
    z1 = np.dot(w1, x_train) + b1
    a1 = tanh(z1)
    a2 = sigmoid(np.dot(w2, a1) + b2)
    dz1 = w2.T * (1 - z1 * z1) * (a2 - y_train)
    dz2 = (a2 - y_train)
    dw1 = np.dot(dz1, x_train.T) / m
    dw2 = np.dot(dz2, a1.T) / m
    db1 = np.sum(dz1, axis = 1) / m
    db2 = np.sum(dz2, axis = 1) / m
    w1 = w1 - alpha * dw1
    w2 = w2 - alpha * dw2
    b1 = b1 - alpha * db1.reshape((4, 1))
    b2 = b2 - alpha * db2.reshape((1, 1))
    a = float(input())
    b = float(input())
    a1 = tanh(np.dot(w1, np.array([a, b]).reshape(2, 1)) + b1)
    a2 = sigmoid(np.dot(w2, a1) + b2)
    print(a2)