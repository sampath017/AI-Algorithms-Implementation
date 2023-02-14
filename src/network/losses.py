import numpy as np


class Loss:
    def calculate(self, output, y_true):
        sample_losses = self.forward(output, y_true)
        data_loss = np.mean(sample_losses)

        return data_loss


class CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        n_samples = y_pred.shape[0]

        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if y_true.ndim == 1:
            correct_confidences = y_pred_clipped[range(n_samples), y_true]
        elif y_true.ndim == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods

    def backward(self, y_pred, y_true):
        n_samples = len(y_pred)
        n_labels = len(y_pred[0])

        if y_true.ndim == 1:
            y_true = np.eye(n_labels)[y_true]

        self.dinputs = -y_true / y_pred
        self.dinputs = self.dinputs / n_samples
