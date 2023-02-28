import numpy as np


class ActivationSoftmaxLossCategoricalCrossentropy:
    def backward(self, y_pred, y_true):
        n_samples = len(y_pred)

        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = y_pred.copy()
        self.dinputs[range(n_samples), y_true] -= 1
        self.dinputs = self.dinputs / n_samples
