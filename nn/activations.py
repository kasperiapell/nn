import numpy as np
from numpy import *

def RELU(t):
    return np.maximum(t, 0)
    
def RELU_grad(t):
    x = t * (t > 0)
    y = ma.divide(x, x)
    return y.data

def GELU(t):
    A = 0.5 * t
    B = np.sqrt(2 / np.pi)
    C = t + 0.044715 * np.power(t, 3)

    return A * (1 + np.tanh(B * C))

def phi(x):
    A = 1 / np.sqrt(2 * np.pi)
    B = -0.5 * np.power(x, 2)

    return A * np.exp(B)

def GELU_grad(t):
    A = np.sqrt(2 / np.pi)
    B = t + 0.044715 * np.power(t, 3)
    C = 0.5 * (1 + np.tanh(A * B))

    return C - t * phi(t)

def softmax(eta):
    offset = np.array([np.max(eta, axis = 1)])
    X = np.sum(np.exp(eta - offset.T), axis = 1)
    X = ma.log(X) + offset
    X = X.T.filled(0)

    return np.exp(eta - X)

def softmax_grad(eta):
    matrices = np.zeros((eta.shape[0], eta.shape[1], eta.shape[1]))
    for i in range(eta.shape[0]):
        x = np.array([eta[i]])
        offset = np.max(x)
        y = np.exp(x)
        A = np.dot(y.T, y)
        B = np.sum(np.exp(x - offset), axis = 1)
        B = ma.log(Q) + offset
        B = B.T.filled(0)

        X = -1 * (A * np.exp(- 2 * B))  
        Y = softmax(x)
        
        np.fill_diagonal(X, Y + np.diag(X))
            
        matrices[i] = X
        
    return matrices