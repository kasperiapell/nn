import unittest

class TestFunctions(unittest.TestCase):
    def test_softmax(self):
        neural_network = NeuralNetwork()
        eta0 = np.array([[0, 3/2, 1], [1, 1, 1]])
        eta1 = np.array([[1, 2], [0.5, 0.5], [0, 5]])
        S0 = np.array([[0.12195165, 0.54654939, 0.33149896], [1/3, 1/3, 1/3]])
        S1 = np.array([[0.2689414213699951, 0.7310585786300049], 
                       [0.5, 0.5], 
                       [0.0066928509242848554, 0.9933071490757152]])
        
        T0 = neural_network.softmax(eta0)
        T1 = neural_network.softmax(eta1)
        np.testing.assert_array_almost_equal(S0, T0)
        np.testing.assert_array_almost_equal(S1, T1)
        
    def test_softmax_der(self):
        neural_network = NeuralNetwork()
        eta0 = np.array([[1, 2/3], [1, 1]])
        eta1 = np.array([[0, 0, 0], [0.1, 0.2, 0.3], [30, 15, 50]])
        
        S0 = np.array([[[ 0.24318216, -0.24318216],
                        [-0.24318216,  0.24318216]], 
                          [[ 0.25, -0.25],
                           [-0.25,  0.25]]])
        
        T0 = neural_network.softmax_der(eta0)
        np.testing.assert_array_almost_equal(S0, T0)

    def test_loss(self):
        nn = NeuralNetwork()
        probs0 = np.array([[0.5, 0.5]])
        probs1 = np.array([[0.2, 0, 0.15, 0.65], [0.1, 0.1, 0.1, 0.7]])
        y0 = np.array([[1]])
        y1 = np.array([[2, 0]])
        
        S1 = np.array([[1.8971199848858813], [2.3025850929940455]])
        
        np.testing.assert_almost_equal(nn.loss(probs0, y0), 0.6931471805599453)
        #np.testing.assert_almost_equal(nn.loss(probs1, y1), S1)
      
    def test_loss_derivative(self):
        nn = NeuralNetwork(())
        probs0 = np.array([[0.2, 0.8]])
        probs1 = np.array([[0.1, 0.2, 0.3, 0.4]])
        y0 = np.array([[1, 0]])
        y1 = np.array([[0, 0, 0, 1]])
        
        a = np.array([[0.6213349345596119, 0]])
        b = np.array([[0, 0, 0, 1.0913566679372915]])
        np.testing.assert_almost_equal(nn.loss_derivative(probs0, y0), a)
        np.testing.assert_almost_equal(nn.loss_derivative(probs1, y1), b)