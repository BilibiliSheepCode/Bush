import sys
import os
import numpy as np
import math
import time

'''
#def y_h = sigmoid(w ^ T * x + b) WHILE sigmoid(z) = 1 / (1 + e^(-z)) ; y_h MEANS y_hat

#def L(y_h, y) = -(y * log(y_h) + (1 - y) * log(1 - y_h)) AS Loss_Function

#def J(w, b) = (1 / m) * SIGMA(i = 1, m)(L(y_h ^ (i), y ^ (i))) AS Cost_Function

#def Gradient_Descent: {
    REPEAT {
        w := w - .alpha. * d J(w, b) / d w ; d EQU partial d AT HERE
        b := b - .alpha. * d J(w, b) / d b ; d EQU partial d AT HERE
    }
}

'''

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

m = 0
n_x = 2
dw = np.zeros((n_x, 1))