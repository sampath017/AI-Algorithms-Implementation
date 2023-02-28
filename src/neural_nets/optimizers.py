import numpy as np


class SGD:
    def __init__(self, lr=1.0, lr_decay=0.0, momentum=0.0):
        self.lr = lr
        self.current_lr = lr
        self.lr_decay = lr_decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.lr_decay:
            self.current_lr = self.lr / (1.0 + self.lr_decay * self.iterations)

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, "weights_momentums"):
                layer.weights_momentums = np.zeros_like(layer.weights)
                layer.biases_momentums = np.zeros_like(layer.biases)

            weights_updates = layer.weights_momentums * \
                self.momentum - layer.dweights * self.current_lr
            layer.weights_momentums = weights_updates
            biases_updates = layer.biases_momentums * \
                self.momentum - layer.dbiases * self.current_lr
            layer.biases_momentums = biases_updates

        else:
            weights_updates = -self.current_lr * layer.dweights
            biases_updates = -self.current_lr * layer.dbiases

        layer.weights += weights_updates
        layer.biases += biases_updates

    def post_update_params(self):
        self.iterations += 1


class Adam:

    def __init__(self, lr=0.001, lr_decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.lr = lr
        self.current_lr = lr
        self.lr_decay = lr_decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.lr_decay:
            self.current_lr = self.lr / (1. + self.lr_decay * self.iterations)

    def update_params(self, layer):
        if not hasattr(layer, 'weights_cache'):
            layer.weights_momentums = np.zeros_like(layer.weights)
            layer.weights_cache = np.zeros_like(layer.weights)
            layer.biases_momentums = np.zeros_like(layer.biases)
            layer.biases_cache = np.zeros_like(layer.biases)

        layer.weights_momentums = self.beta_1 * \
            layer.weights_momentums + (1 - self.beta_1) * layer.dweights
        layer.biases_momentums = self.beta_1 * \
            layer.biases_momentums + (1 - self.beta_1) * layer.dbiases

        weights_momentums_corrected = layer.weights_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.biases_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))

        layer.weights_cache = self.beta_2 * layer.weights_cache + \
            (1 - self.beta_2) * layer.dweights**2
        layer.biases_cache = self.beta_2 * layer.biases_cache + \
            (1 - self.beta_2) * layer.dbiases**2

        weights_cache_corrected = layer.weights_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        biases_cache_corrected = layer.biases_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_lr * \
            weights_momentums_corrected / \
            (np.sqrt(weights_cache_corrected) + self.epsilon)
        layer.biases += -self.current_lr * \
            bias_momentums_corrected / \
            (np.sqrt(biases_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
