import numpy as np


def Accuracy(y_pred, y_true):
    predictions = np.argmax(y_pred, axis=1)
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)

    return np.mean(predictions == y_true)
