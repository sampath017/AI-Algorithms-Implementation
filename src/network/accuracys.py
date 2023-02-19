import numpy as np


class Accuracy:
    def calculate(self, y_pred, y_true):
        comparisons = self.compare(y_pred, y_true)
        accuracy = np.mean(comparisons)

        return accuracy


class Regression(Accuracy):
    def __init__(self):
        self.precison = None

    def init(self, y_true, reinit=False):
        if self.precison is None or reinit:
            self.precison = np.std(y_true) / 250

    def compare(self, y_pred, y_true):
        return np.abs(y_pred - y_true) < self.precison


class Categorical(Accuracy):
    def __init__(self, *, binary=False):
        self.binary = binary

    def init(self, y_true):
        pass

    def compare(self, y_pred, y_true):
        if not self.binary and y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)

        return y_pred == y_true
