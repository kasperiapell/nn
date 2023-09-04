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
    mu = np.array([np.max(eta, axis = 1)])
    X = np.sum(np.exp(eta - mu.T), axis = 1)
    Y = ma.log(X) + mu
    Z = Y.T.filled(0)

    return np.exp(eta - Z)

def softmax_grad(eta):
    matrices = np.zeros((eta.shape[0], eta.shape[1], eta.shape[1]))
    for i in range(eta.shape[0]):
        x = np.array([eta[i]])
        mu = np.max(x)
        y = np.exp(x)
        A = np.dot(y.T, y)
        B = np.sum(np.exp(x - mu), axis = 1)
        C = ma.log(B) + mu
        D = C.T.filled(0)

        X = -1 * (A * np.exp(- 2 * D))  
        E = softmax(x)
        
        np.fill_diagonal(X, E + np.diag(X))
            
        matrices[i] = X
        
    return matrices