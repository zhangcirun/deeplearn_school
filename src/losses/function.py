from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, y_pred, y_true):
        """
        :param y_pred: 2D prediction batches numpy array
        :param y_true: true values
        :return: losses value
        """
        raise NotImplementedError

    def get_loss(self, y_pred, y_true):
        """
        Get mean losses
        :param y_pred: 2D prediction batches numpy array
        :param y_true: true values
        :return: mean losses value
        """
        losses = self.forward(y_pred, y_true)
        mean_loss = np.mean(losses)
        return mean_loss


class CrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        """
        Cross entropy losses
        :param y_pred: 2D prediction batches numpy array
        :param y_true: 1D numeric labels or 2D one-hot labels
        :return: losses value
        """
        y_pred = np.clip(y_pred, 1e-7, 1-1e-7)  # prevent confidence from being exactly 0
        if y_true.ndim == 1:  # numeric categorical labels
            confidences = y_pred[range(len(y_pred)), y_true]
        elif y_true.ndim == 2:  # one-hot labels
            confidences = np.sum(y_pred * y_true, axis=1)
        else:
            raise Exception
        return -np.log(confidences)  # averaged losses per batch


if __name__ == '__main__':
    softmax_outputs = np.array([[0.7, 0.1, 0.2],
                                [0.1, 0.5, 0.4],
                                [0.02, 0.9, 0.08]])
    class_targets = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 1, 0]])

    class_targets2 = np.array([0, 1, 1])

    print(CrossEntropy().get_loss(softmax_outputs, class_targets))
