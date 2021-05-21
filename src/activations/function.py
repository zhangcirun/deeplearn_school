from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, arr):
        raise NotImplementedError


class Linear(Activation):
    def forward(self, arr):
        return arr


class Relu(Activation):
    def forward(self, arr):
        return np.maximum(0.0, arr)


class Sigmoid(Activation):
    def forward(self, arr):
        return 1 / (1 + np.exp(-arr))


class Softmax(Activation):
    def forward(self, arr):
        tmp = np.exp(arr) - np.max(arr, axis=1, keepdims=True)  # handle the overflowing issue
        return tmp / np.sum(tmp, axis=1, keepdims=True)
