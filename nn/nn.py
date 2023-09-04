import numpy as np
from numpy import *

class Layer():
    def __init__(self, identifier, input_layer, output_layer, 
                 size, weight, bias, prev_layer, next_layer):
        self.id = identifier
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.size = size
        self.input = None
        self.output = None
        self.W = weight
        self.b = bias
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        self.W_gradient = weight
        self.b_gradient = bias
        self.gradient = None

class NeuralNetwork():
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.layer_count = len(layer_sizes)
        self.layers = []
        self.input_layer = None
        self.output_layer = None

        self.initialize_layers()

    def initialize_layers(self):
        for i in range(layer_count):
            input_layer = (i == 0)

            if i == self.layer_count - 1:
                output_layer = 1
                weight = None
                bias = None
            else:
                output_layer = False
                a = layer_sizes[i + 1]
                b = layer_sizes[i]
                weight = np.random.uniform(-1, 1, (a, b))
                bias = np.random.uniform(-1, 1, (a, 1))

            layer = Layer(identifier = i,
                          input_layer = input_layer, 
                          output_layer = output_layer, 
                          size = self.layer_sizes[i], 
                          weight = weight, 
                          bias = bias, 
                          prev_layer = None, 
                          next_layer = None)

            self.layers.append(layer)

        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

        for i in range(layer_count - 1):
            self.layers[i].next_layer = self.layers[i + 1]
            if i > 0:
                self.layers[i].prev_layer = self.layers[i - 1]
   
    def inspect_weights(self):
        print("Weights: ")
        print(self.W)
        print("Biases: ")
        print(self.bias)
        
    def softmax(self, eta):
        m = np.array([np.max(eta, axis = 1)])
        S = np.sum(np.exp(eta - m.T), axis = 1)
        Sma = ma.log(S) + m
        B = Sma.T.filled(0)

        return np.exp(eta - B)
    
    def softmax_der(self, eta):
        matrices = np.zeros((eta.shape[0], eta.shape[1], eta.shape[1]))
        for i in range(eta.shape[0]):
            x = np.array([eta[i]])
            m = np.max(x)
            z = np.exp(x)
            Q = np.dot(z.T, z)
            S = np.sum(np.exp(x - m), axis = 1)
            Sma = ma.log(S) + m
            B = Sma.T.filled(0)

            matr = -1 * (Q * np.exp(- 2 * B))  
            N = self.softmax(x)
        
            np.fill_diagonal(matr, N + np.diag(matr))
            
            matrices[i] = matr
        
        return matrices
        
    def act(self, x):
        return np.maximum(x, 0)
    
    def act_der(self, x):
        a = x * (x > 0)
        p = ma.divide(a, a)
        q = p.data
        return q
    
    def forward(self, X):
        L = self.input_layer
        L.input = X
        eta = np.dot(X, L.W.T)
        eta += tile(L.b.T, (eta.shape[0], 1))
        L.output = self.act(eta)
        
        for i in range(1, len(self.layers)):
            H = self.layers[i - 1].output
            layer = self.layers[i]
            layer.input = H
            if layer.output_layer == True:
                layer.output = self.softmax(H)
            else:
                eta = np.dot(H, layer.W.T)
                eta += tile(layer.b.T, (eta.shape[0], 1))
                layer.output = self.act(eta)
    
    def predict(self, X):
        self.forward(X)
        probabilities = self.output_layer.output
        return np.argmax(probabilities, axis = 1)
    
    def probabilities(self, X):
        self.forward(X)
        return self.output_layer.output
    
    def loss(self, Y):
        probs = self.output_layer.output
        ind = np.zeros((probs.shape[0], probs.shape[1]))
        seq = list(range(probs.shape[0]))
        ind[(seq, Y.astype(int).T)] = 1
        pred_probs = probs[ind == 1]
        
        pp = ma.log(pred_probs)
        
        return -1 * np.sum(pp.filled(0))
    
    def loss_derivative(self, Y):
        probs = self.output_layer.output
        ind = np.zeros((probs.shape[0], probs.shape[1]))
        seq = list(range(probs.shape[0]))
        ind[(seq, Y.astype(int))] = 1
        pred_probs = probs * ind
        
        pp = ma.log(pred_probs)   
        pd = ma.divide(1, pp)     
        self.output_layer.gradient = -1 * pd.filled(0)
    
    def backward(self, X, Y):
        self.loss_derivative(Y)
        for i in range(len(self.layers) - 2, -1, -1):
            self.update_gradients(i)
            
    def update_gradients(self, layer_id):
        layer = self.layers[layer_id]
        prev_layer = layer.prev_layer
        next_layer = layer.next_layer

        X = layer.input        
        dLdZ = next_layer.gradient
        eta = next_layer.input
        dzdeta = [self.act_der(np.array([e])) for e in eta]
        detadW = np.zeros((X.shape[0], next_layer.size, layer.size, next_layer.size))
                          
        for i in range(X.shape[0]):            
            temp = np.zeros((next_layer.size, layer.size, next_layer.size))

            for j in range(next_layer.size):
                temp[j][:,j] = X[i,:]

            detadW[i] = temp
                          
        dLdb = np.zeros((X.shape[0], next_layer.size, 1))
        dLdW = np.zeros((X.shape[0], layer.size, next_layer.size))
        for i in range(X.shape[0]):
            vec = np.sum(dLdZ[i] * dzdeta[i].T, axis = 1)
            dLdb[i] = np.array([vec]).T
            for j in range(detadW[i].shape[0]):
                dLdW[i] += vec[j] * detadW[i][j]
                
        detady = np.zeros((X.shape[0], next_layer.size, layer.size))
        for i in range(X.shape[0]):
            detady[i] = layer.W
            
        dLdy = np.zeros((X.shape[0], layer.size))
        for i in range(X.shape[0]):
            vec = np.sum(dLdZ[i] * dzdeta[i].T, axis = 1)
            dLdy[i] = np.dot(vec, detady[i])
                   
        layer.W_gradient = dLdW
        layer.b_gradient = dLdb
        layer.gradient = dLdy
    
    def update_weights(self, alpha):
        for layer in self.layers[:-1]:
            layer.W -= alpha * np.mean(layer.W_gradient, axis = 0).T
            layer.b -= alpha * np.mean(layer.b_gradient, axis = 0)
    
    def train(self, train_X, train_Y, alpha, epochs):
        for iteration in range(epochs):
            self.forward(train_X)
            self.backward(train_X, train_Y)
            self.update_weights(0.001)
            
            loss = self.loss(train_Y)
            
            if iteration % 2 == 0:
                print(loss)
                
        print("Finished training")