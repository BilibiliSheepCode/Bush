import time
import numpy as np
from PIL import Image
from scipy import ndimage

def normalizeRows(x):
    x = x / np.linalg.norm(x, axis = 1, keepdims = True)
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x[1] * (1 - x[1])

def softmax(x):
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis = 1, keepdims = True)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1-(x[1] * x[1])

def ReLU(x):
    return np.max(0, x)

def d_ReLU(x):
    return np.where(x[0] < 0, 0, 1)

def Leaky_ReLU(x):
    return np.max(0.01 * x, x)

def L1(yhat, y):
    return np.sum(np.abs(y - yhat))

def L2(yhat, y):
    return np.dot((y - yhat),(y - yhat).T)

def preproving(param, caches, hyperparam):
    L = hyperparam["L"]
    for l in range(1, L + 1):
        sl = str(l)
        caches["Z" + sl] = np.dot(param["W" + sl], caches["A" + str(l - 1)]) + param["B" + sl]
        caches["A" + sl] = hyperparam['functions'][hyperparam["g"][l - 1]](caches["Z" + sl])
    return caches

def backpropagate(param, caches, hyperparam):
    L = hyperparam["L"]
    m = hyperparam["m"]
    caches["dZ" + str(L)] = caches['A' + str(L)] - caches['Y']
    caches["dW" + str(L)] = np.dot(caches["dZ" + str(L)], caches["A" + str(L - 1)].T) / m
    caches["dB" + str(L)] = np.sum(caches["dZ" + str(L)], axis = 1, keepdims = True) / m
    for l in range(L - 1, 0, -1):
        sl = str(l)
        caches["dA" + sl] = np.dot(param["W" + str(l + 1)].T, caches["dZ" + str(l + 1)])
        caches["dZ" + sl] = caches["dA" + sl] * hyperparam['functions']['d_' + hyperparam["g"][l - 1]]([caches["Z" + sl], caches["A" + sl]])
        caches["dW" + sl] = np.dot(caches["dZ" + sl], caches["A" + str(l - 1)].T) / m
        caches["dB" + sl] = np.sum(caches["dZ" + sl], axis = 1, keepdims = True) / m
    return caches

def update(param, caches, hyperparam):
    L = hyperparam["L"]
    for l in range(1, L + 1):
        sl = str(l)
        param["W" + sl] -= hyperparam["learningrate"] * caches["dW" + sl]
        param["B" + sl] -= hyperparam["learningrate"] * caches["dB" + sl]
    return param

def init(layerdims, hyperparam):
    L = hyperparam["L"]
    param = {}
    for l in range(L):
        param["W" + str(l + 1)] = np.random.randn(layerdims[l + 1], layerdims[l]) * 0.01
        param["B" + str(l + 1)] = np.zeros((layerdims[l + 1], 1))
    return param

m = 1000000
alpha = 0.1
num_it = 1000
hyperparam = {
    "learningrate": alpha,
    "m": m,
    'L': 2,
    'g': ['tanh','sigmoid'],
    'functions': {
        'tanh': tanh,
        'd_tanh': d_tanh,
        'sigmoid': sigmoid,
        'd_sigmoid': d_sigmoid,
        'relu': ReLU,
        'd_relu': d_ReLU,
        'softmax': softmax
    }
}
param = init([2, 4, 1], hyperparam)
while True:
    x_train = np.random.rand(2, m)
    y_train = np.where(x_train[0] > x_train[1], 0, 1)
    caches = {
        'A0': x_train,
        'Y': y_train
    }
    for i in range(num_it):
        caches = preproving(param, caches, hyperparam)
        caches = backpropagate(param, caches, hyperparam)
        param = update(param, caches, hyperparam)
    loss = np.sum(L2(caches['A2'], y_train)) / m
    a = float(input('a'))
    b = float(input('b'))
    caches['A0'] = np.array([a,b]).reshape((2,1))
    caches = preproving(param, caches, hyperparam)
    a2 = caches['A2']
    print(f'a:{a} b:{b}\na < b: {1 if a2 > 0.5 else 0}\nrisk:{a2} loss:{loss}\n')