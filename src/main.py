import nnfs
from nnfs.datasets import spiral_data, sine_data

import numpy as np
import matplotlib.pyplot as plt

from network.layers import Dense, Dropout
from network.activations import ReLU, Sigmoid, Linear
from network.commons import ActivationSoftmaxLossCategoricalCrossentropy
from network.optimizers import Adam
from network.losses import BinaryCrossentropy, MeanSquaredError

nnfs.init()

X, y = sine_data()

dense1 = Dense(1, 64)
activation1 = ReLU()
dense2 = Dense(64, 64)
activation2 = ReLU()
dense3 = Dense(64, 1)
activation3 = Linear()
loss_function = MeanSquaredError()
optimizer = Adam(lr=0.005, lr_decay=1e-3)

accuracy_precision = np.std(y) / 250

# Training model
EPOCHS = 10_000
for epoch in range(EPOCHS+1):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    data_loss = loss_function.calculate(activation3.output, y)
    regularization_loss = loss_function.regularization_loss(dense1) +\
        loss_function.regularization_loss(dense2) +\
        loss_function.regularization_loss(dense3)

    loss = data_loss + regularization_loss

    # Metrics
    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) <
                       accuracy_precision)

    if epoch % 100 == 0:
        print(
            f"epoch: {epoch} acc: {accuracy:.3f} (data_loss: {data_loss:.3f} regularization_loss: {regularization_loss:.3f}) loss: {loss}  lr: {optimizer.current_lr:.3f}")

    # Backpass
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
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
X_test, y_test = sine_data()

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)
loss = loss_function.calculate(activation3.output, y_test)

# Metrics
predictions = activation3.output
accuracy = np.mean(np.absolute(predictions - y) <
                   accuracy_precision)

print("\nTesting Metrics")
print(f"acc: {accuracy:.3f} loss: {loss:.3f}\n")

plt.plot(X_test, y_test)
plt.plot(X_test, activation3.output)
plt.show()
