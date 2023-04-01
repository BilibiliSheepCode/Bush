import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
import struct
import sys

def normalizeRows(x):
    x = x / np.linalg.norm(x, axis = 1, keepdims = True)
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x[1] * (1 - x[1])

def softmax(x):
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis = 0, keepdims = True)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1-(x[1] * x[1])

def ReLU(x):
    return np.maximum(0, x)

def d_ReLU(x):
    return np.where(x[0] < 0, 0, 1)

def Leaky_ReLU(x):
    return np.max(0.01 * x, x)

def L1(yhat, y):
    return np.sum(np.abs(y - yhat))

def L2(yhat, y):
    return np.dot((y - yhat),(y - yhat).T)

def dis(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def d_L(a, y):
    return -(y / a) + (1 - y) / (1 - a)

def preproving(param, caches, hyperparam):
    L = hyperparam["L"]
    for l in range(1, L + 1):
        sl = str(l)
        caches["Z" + sl] = np.dot(param["W" + sl], caches["A" + str(l - 1)]) + param["B" + sl]
        caches["A" + sl] = hyperparam["functions"][hyperparam["g"][l - 1]](caches["Z" + sl])
    return caches

def backpropagate(param, caches, hyperparam):
    L = hyperparam["L"]
    m = hyperparam["m"]
    caches["dZ" + str(L)] = caches["A" + str(L)] - caches["Y"]
    caches["dW" + str(L)] = np.dot(caches["dZ" + str(L)], caches["A" + str(L - 1)].T) / m
    caches["dB" + str(L)] = np.sum(caches["dZ" + str(L)], axis = 1, keepdims = True) / m
    for l in range(L - 1, 0, -1):
        sl = str(l)
        caches["dA" + sl] = np.dot(param["W" + str(l + 1)].T, caches["dZ" + str(l + 1)])
        caches["dZ" + sl] = caches["dA" + sl] * hyperparam["functions"]["d_" + hyperparam["g"][l - 1]]([caches["Z" + sl], caches["A" + sl]])
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

def init(hyperparam):
    L = hyperparam["L"]
    layerdims = hyperparam["layerdims"]
    layerdims.insert(0, hyperparam["nx"])
    param = {}
    for l in range(L):
        param["W" + str(l + 1)] = np.random.randn(layerdims[l + 1], layerdims[l]) * 0.01
        param["B" + str(l + 1)] = np.zeros((layerdims[l + 1], 1))
    return param

def load_minst(path, kind = "train"):
    labels_path = os.path.join(path,"%s-labels.idx1-ubyte" % kind)
    images_path = os.path.join(path,"%s-images.idx3-ubyte" % kind)
    with open(labels_path, "rb") as lbpath:
        magic, n = struct.unpack(">II", lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images.T, labels

def load(path):
    y_trainu = [[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],
            [0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]]
    x_train, y_trainf = load_minst(path)
    y_train = np.zeros((y_trainf.shape[0], 10))
    for i in range(y_trainf.shape[0]):
        y_train[i] = np.array(y_trainu[y_trainf[i]])
    return x_train, y_train.T

x_train, y_train = load("./mnist")

hyperparam = {
    "learningrate": 0.1,
    "m": x_train.shape[1],
    "L": 2,
    "nx": 784,
    "layerdims": [112, 10],
    "g": ["tanh", "softmax"],
    "functions": {
        "tanh": tanh,
        "d_tanh": d_tanh,
        "sigmoid": sigmoid,
        "d_sigmoid": d_sigmoid,
        "relu": ReLU,
        "d_relu": d_ReLU,
        "softmax": softmax
    }
}

L = hyperparam["L"]
sL = str(L)

param = init(hyperparam)

caches = {
    "A0": x_train,
    "Y": y_train
}

if os.path.exists(".\\Params"):
    que = input("Init with last trained param? Y or N")
    if que == "Y" or que == 'y':
        for l in range(1, L + 1):
            sl = str(l)
            d = np.load(".\\Params\\Param" + sl + ".npz")
            param["W" + sl], param["B" + sl] = d["arr_0"], d["arr_1"]

que = int(input("Train or test? 0 or 1"))
if que == 1:
    while True:
        test = np.array(Image.open(".\\assets\\images\\test.png").convert("L"))
        caches["A0"] = test.reshape((hyperparam["nx"], 1))
        caches = preproving(param, caches, hyperparam)
        print(f"\nPredict: {np.argmax(caches['A' + sL])}")
        print(f"Rates: \n{caches['A' + sL]}")
        exi = input("Input q to exit or enter to retesting.")
        if exi == 'q' or exi == 'Q':
            break
    sys.exit(0)

while True:
    caches["A0"] = x_train
    caches["Y"] = y_train
    for i in range(1000):
        caches = preproving(param, caches, hyperparam)
        caches = backpropagate(param, caches, hyperparam)
        param = update(param, caches, hyperparam)
    print(f"cost: {L1(caches['A' + sL], caches['Y'])}")
    test = np.array(Image.open(".\\assets\\images\\test.png").convert("L"))
    caches["A0"] = test.reshape((hyperparam["nx"], 1))
    caches = preproving(param, caches, hyperparam)
    print(f"\nPredict: {np.argmax(caches['A' + sL])}")
    print(f"Rates: \n{caches['A' + sL]}")
    exi = input("Input q to exit or enter to continue training.")
    if exi == 'q' or exi == 'Q':
        break

for l in range(1, L + 1):
    sl = str(l)
    np.savez(".\\Params\\Param" + sl,param["W" + sl], param["B" + sl])