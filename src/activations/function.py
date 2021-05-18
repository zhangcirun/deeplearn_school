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
