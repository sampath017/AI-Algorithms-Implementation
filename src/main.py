import nnfs
from nnfs.datasets import spiral_data

import numpy as np

from network.layers import Dense, Dropout
from network.activations import ReLU, Sigmoid
from network.commons import ActivationSoftmaxLossCategoricalCrossentropy
from network.optimizers import Adam
from network.losses import BinaryCrossentropy
from network.metrics import Accuracy

nnfs.init()

X, y = spiral_data(samples=100, classes=2)
y = y.reshape(-1, 1)

dense1 = Dense(2, 64, l2_weight_regularizer=5e-4, l2_bias_regularizer=5e-4)
activation1 = ReLU()
dense2 = Dense(64, 1)
activation2 = Sigmoid()
loss_function = BinaryCrossentropy()
optimizer = Adam(lr_decay=5e-7)

# Training model
EPOCHS = 10_000
for epoch in range(EPOCHS+1):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    data_loss = loss_function.calculate(activation2.output, y)
    regularization_loss = loss_function.regularization_loss(
        dense1) + loss_function.regularization_loss(dense2)
    loss = data_loss + regularization_loss

    # Metrics
    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions == y)

    if epoch % 100 == 0:
        print(
            f"epoch: {epoch} acc: {accuracy:.3f} (data_loss: {data_loss:.3f} regularization_loss: {regularization_loss:.3f}) loss: {loss:.3f}  lr: {optimizer.current_lr:.3f}")

    # Backpass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update params
    optimizer.pre_update_params()
    optimizer.update_params(dense2)
    optimizer.update_params(dense1)
    optimizer.post_update_params()

# Testing model
# Forward pass
X_test, y_test = spiral_data(samples=100, classes=2)
y_test = y_test.reshape(-1, 1)

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss = loss_function.calculate(activation2.output, y_test)

# Metrics
predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions == y_test)

print("\nTesting Metrics")
print(f"acc: {accuracy:.3f} loss: {loss:.3f}\n")
