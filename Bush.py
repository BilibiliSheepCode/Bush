import sys
import os
import math
import time
import numpy as np
from PIL import Image
from scipy import ndimage

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def image2vector(image):
    return image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)

def normalizeRows(x):
    x = x / np.linalg.norm(x, axis = 1, keepdims = True)
    return x

def softmax(x):
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis = 1, keepdims = True)

def tanh(x):
    ex = np.exp(x)
    e_x = np.exp(-x)
    return (ex - e_x) / (ex + e_x)

def ReLU(x):
    return np.max(0, x)

def Leaky_ReLU(x):
    return np.max(0.01 * x, x)

def L1(yhat, y):
    return np.sum(np.abs(y - yhat))

def L2(yhat, y):
    return np.dot((y - yhat),(y - yhat).T)

