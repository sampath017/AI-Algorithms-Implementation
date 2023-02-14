import numpy as np
from .activations import Softmax
from .losses import CategoricalCrossentropy


class ActivationSoftmaxLossCategoricalCrossentropy:
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)

    def backward(self, y_pred, y_true):
        n_samples = len(y_pred)

        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = y_pred.copy()
        self.dinputs[range(n_samples), y_true] -= 1
        self.dinputs = self.dinputs / n_samples
