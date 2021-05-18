import numpy as np

from activations.function import *


class Dense:
    """
    :keyword self.weights: m x n weights matrix of neurons, each column is the weight vector for a neuron
    :keyword self.biases: 1 x n vector of biases
    :keyword self.outputs: 2D n_batch x n_neurons output matrix, each row is the output vector for a single batch
    """

    def __init__(self, n_input, n_neurons):
        """

        :param n_input: length of a single input vector (m)
        :param n_neurons: number of neurons in the layer (n)
        """
        self.weights = 0.1 * np.random.randn(n_input, n_neurons)  # weights initialization
        self.biases = np.zeros((1, n_neurons))  # biases initialization

    def forward(self, inputs, activations='relu'):
        """
        :param inputs: input batches matrix, each row is a input vector for a single batch
        :param activations: activation function, default='relu'
        """
        self.output = np.dot(inputs, self.weights) + self.biases

        if activations == 'linear':
            self.output = Linear().forward(self.output)
        elif activations == 'relu':
            self.output = Relu().forward(self.output)
        elif activations == 'sigmoid':
            self.output = Sigmoid().forward(self.output)
        else:
            raise Exception("Invalid activation function")


if __name__ == '__main__':
    X = np.array([[1, 2, 3, 2.5],  # input1
                  [2, 5, -1, 2],  # input2
                  [-1.5, 2.7, 3.3, -0.8]])  # input3

    layer1 = Dense(4, 5)
    layer2 = Dense(5, 2)
    layer1.forward(X, 'sigmoid')
    layer2.forward(layer1.output, 'relu')

