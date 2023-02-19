import numpy as np


class Loss:
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y_true, include_regularization=False):
        sample_losses = self.forward(output, y_true)
        data_loss = np.mean(sample_losses)

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()

    def regularization_loss(self):
        regularization_loss = 0

        for layer in self.trainable_layers:
            if layer.l1_weight_regularizer > 0:
                regularization_loss += layer.l1_weight_regularizer * \
                    np.sum(abs(layer.weights))

            if layer.l1_bias_regularizer > 0:
                regularization_loss += layer.l1_bias_regularizer * \
                    np.sum(abs(layer.biases))

            if layer.l2_weight_regularizer > 0:
                regularization_loss += layer.l2_weight_regularizer * \
                    np.sum(layer.weights ** 2)

            if layer.l2_bias_regularizer > 0:
                regularization_loss += layer.l2_bias_regularizer * \
                    np.sum(layer.biases ** 2)

        return regularization_loss


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


class BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    def backward(self, y_pred, y_true):

        n_samples = y_pred.shape[0]
        n_outputs = y_pred.shape[1]

        clipped_y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / clipped_y_pred -
                         (1 - y_true) / (1 - clipped_y_pred)) / n_outputs
        self.dinputs = self.dinputs / n_samples


class MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

        return sample_losses

    def backward(self, y_pred, y_true):
        n_samples = y_pred.shape[0]
        n_outputs = y_pred.shape[1]

        self.dinputs = -2 * (y_true - y_pred) / n_outputs
        self.dinputs = self.dinputs / n_samples


class MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        return sample_losses

    def backward(self, y_pred, y_true):
        n_samples = y_pred.shape[0]
        n_outputs = y_pred.shape[1]

        self.dinputs = -np.sign(y_true - y_pred) / n_outputs
        self.dinputs = self.dinputs / n_samples
