import numpy as np

if __name__ == '__main__':
    a = np.array([[1, 2, 3, 2.5],  # input1
                  [2, 5, -1, 2],  # input2
                  [-1.5, 2.7, 3.3, -0.8]])  # input3
    b = np.array([[0.2, 0.8, -0.5, 1],  # neuron1
                  [0.5, -0.91, 0.26, -0.5],  # neuron2
                  [-0.26, -0.27, 0.17, 0.87]])  # neuron3
    print(np.dot(a, b.T))

    print(np.dot(b, a.T))

    print(np.dot(b, a[0]))
