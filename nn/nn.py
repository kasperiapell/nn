import numpy as np
from numpy import *

from activations import *

class Layer():
    def __init__(self, identifier, input_layer, output_layer, 
                 size, weight, bias, prev_layer, next_layer,
                 act, act_grad):
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
        self.act = act
        self.act_grad = act_grad

    @staticmethod
    def inspect_weights(self):
        print("Layer ", self.id)
        if self.output_layer:
            print("Output layer")
        else:
            print("Weight:")
            print(self.W)
            print("Bias:")
            print(self.b)

    @staticmethod
    def update_gradients(self):
        prev_layer = self.prev_layer
        next_layer = self.next_layer

        X = self.input        
        dLdZ = next_layer.gradient
        eta = next_layer.input
        dZdeta = [self.output_act_grad(np.array([e])) for e in eta]
        detadW = np.zeros((X.shape[0], next_layer.size, self.size, next_layer.size))
                          
        for i in range(X.shape[0]):            
            temp = np.zeros((next_layer.size, self.size, next_layer.size))

            for j in range(next_layer.size):
                temp[j][:,j] = X[i,:]

            detadW[i] = temp
                          
        dLdb = np.zeros((X.shape[0], next_layer.size, 1))
        dLdW = np.zeros((X.shape[0], self.size, next_layer.size))

        for i in range(X.shape[0]):
            vec = np.sum(dLdZ[i] * dZdeta[i].T, axis = 1)
            dLdb[i] = np.array([vec]).T
            for j in range(detadW[i].shape[0]):
                dLdW[i] += vec[j] * detadW[i][j]
                
        detadY = np.zeros((X.shape[0], next_layer.size, self.size))

        for i in range(X.shape[0]):
            detadY[i] = layer.W
            
        dLdY = np.zeros((X.shape[0], self.size))
        for i in range(X.shape[0]):
            vec = np.sum(dLdZ[i] * dZdeta[i].T, axis = 1)
            dLdY[i] = np.dot(vec, detadY[i])
                   
        self.W_gradient = dLdW
        self.b_gradient = dLdb
        self.gradient = dLdY

    def update_weights(self, alpha):
        self.W -= alpha * np.mean(self.W_gradient, axis = 0).T
        self.b -= alpha * np.mean(self.b_gradient, axis = 0)

class NeuralNetwork():
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.layer_count = len(layer_sizes)
        self.layers = []
        self.input_layer = None
        self.output_layer = None
        self.initialize_layers()
        self.initialize_layer_connections()

    @staticmethod
    def initialize_layers(self):
        for i in range(self.layer_count):
            input_layer = (i == 0)

            if i == self.layer_count - 1:
                output_layer = 1
                weight = None
                bias = None
                act = softmax 
                act_grad = softmax_grad
            else:
                output_layer = False
                a = self.layer_sizes[i + 1]
                b = self.layer_sizes[i]
                weight = np.random.uniform(-1, 1, (a, b))
                bias = np.random.uniform(-1, 1, (a, 1))
                act = GELU
                act_grad = GELU_grad

            layer = Layer(identifier = i,
                          input_layer = input_layer, 
                          output_layer = output_layer, 
                          size = self.layer_sizes[i], 
                          weight = weight, 
                          bias = bias, 
                          prev_layer = None, 
                          next_layer = None,
                          act = act,
                          act_grad = act_grad)

            self.layers.append(layer)

    @staticmethod
    def initialize_layer_connections(self):
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

        for i in range(self.layer_count - 1):
            self.layers[i].next_layer = self.layers[i + 1]
            if i > 0:
                self.layers[i].prev_layer = self.layers[i - 1]
   
    @staticmethod
    def inspect_weights(self):
        for layer in self.layers:
            layer.inspect_weights()
            print("\n")
    
    def forward(self, X):
        L = self.input_layer
        L.input = X
        eta = np.dot(X, L.W.T)
        eta += tile(L.b.T, (eta.shape[0], 1))
        L.output = self.act(eta)

        for layer in self.layers[1:]:
            H = layer.prev_layer.output
            layer.input = H

            if layer.output_layer:
                layer.output = self.output_act(H)
            else:
                eta = np.dot(H, layer.W.T)
                eta += tile(layer.b.T, (eta.shape[0], 1))
                layer.output = self.act(eta)
    
    def predict(self, X):
        self.forward(X)
        return np.argmax(self.output_layer.output, axis = 1)
    
    def probabilities(self, X):
        self.forward(X)
        return self.output_layer.output
    
    def loss(self, Y):
        probs = self.output_layer.output
        ind = np.zeros((probs.shape[0], probs.shape[1]))
        seq = list(range(probs.shape[0]))
        ind[(seq, Y.astype(int).T)] = 1
        pred_probs = probs[ind == 1]
        clean_pred_probs = ma.log(pred_probs)
        
        return -np.sum(clean_pred_probs.filled(0))
    
    def loss_derivative(self, Y):
        probs = self.output_layer.output
        ind = np.zeros((probs.shape[0], probs.shape[1]))
        seq = list(range(probs.shape[0]))
        ind[(seq, Y.astype(int))] = 1
        pred_probs = probs * ind
        
        clean_pred_probs = ma.log(pred_probs)   
        clean_pred_probs = ma.divide(1, clean_pred_probs)     
        self.output_layer.gradient = -clean_pred_probs.filled(0)
    
    def backward(self, X, Y):
        self.loss_derivative(Y)
        for layer in self.layers[1:][::-1]
            layer.update_gradients()
    
    def update_weights(self, alpha):
        for layer in self.layers[:-1]:
            layer.update_weights()
    
    def train(self, train_X, train_Y, alpha, epochs):
        for iteration in range(epochs):
            self.forward(train_X)
            self.backward(train_X, train_Y)
            self.update_weights(0.001)
            
            loss = self.loss(train_Y)
            
            if iteration % 2 == 0:
                print(loss)
                
        print("Finished training")