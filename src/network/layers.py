import numpy as np


class Dense:
    def __init__(self, n_inputs, n_neurons,
                 l1_weight_regularizer=0, l2_weight_regularizer=0,
                 l1_bias_regularizer=0, l2_bias_regularizer=0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.l1_weight_regularizer = l1_weight_regularizer
        self.l2_weight_regularizer = l2_weight_regularizer
        self.l1_bias_regularizer = l1_bias_regularizer
        self.l2_bias_regularizer = l2_bias_regularizer

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.l1_weight_regularizer > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.l1_weight_regularizer * dL1

        if self.l1_bias_regularizer > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.l1_bias_regularizer * dL1

        if self.l2_weight_regularizer > 0:
            self.dweights += 2 * self.l2_weight_regularizer * \
                self.weights

        if self.l2_bias_regularizer > 0:
            self.dbiases += 2 * self.l2_bias_regularizer * \
                self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)
