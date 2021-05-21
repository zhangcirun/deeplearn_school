import numpy as np

from activations.function import *


class Dense:
    """
    :keyword self.weights: m x n weights matrix of neurons, each column is the weight vector for a neuron
    :keyword self.biases: 1 x n vector of biases
    :keyword self.outputs: 2D n_batch x n_neurons output matrix, each row is the output vector for a single batch
    :keyword self.activation: activation function object
    """

    def __init__(self, n_input, n_neurons, activation='linear'):
        """

        :param n_input: length of a single input vector (m)
        :param n_neurons: number of neurons in the layer (n)
        :param activation: activation function
        """
        self.weights = 0.1 * np.random.randn(n_input, n_neurons)  # weights initialization
        self.biases = np.zeros((1, n_neurons))  # biases initialization

        if activation == 'linear':
            self.activation = Linear()
        elif activation == 'relu':
            self.activation = Relu()
        elif activation == 'sigmoid':
            self.activation = Sigmoid()
        elif activation == 'softmax':
            self.activation = Softmax()
        else:
            raise Exception("Invalid activation function")

    def forward(self, inputs):
        """
        :param inputs: input batches matrix, each row is a input vector for a single batch
        """
        self.output = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation.forward(self.output)


if __name__ == '__main__':
    X = np.array([[1, 2, 3, 2.5],  # input1
                  [2, 5, -1, 2],  # input2
                  [-1.5, 2.7, 3.3, -0.8]])  # input3

    layer1 = Dense(4, 5, 'relu')
    layer2 = Dense(5, 2, 'softmax')
    layer1.forward(X)
    layer2.forward(layer1.output)

