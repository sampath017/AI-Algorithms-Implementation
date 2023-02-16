import nnfs
from nnfs.datasets import spiral_data

from network.layers import Dense
from network.activations import ReLU
from network.commons import ActivationSoftmaxLossCategoricalCrossentropy
from network.optimizers import Adam
from network.metrics import Accuracy

nnfs.init()

X, y = spiral_data(100, 3)

dense1 = Dense(2, 64)
activation1 = ReLU()
dense2 = Dense(64, 3)
activation_loss = ActivationSoftmaxLossCategoricalCrossentropy()
optimizer = Adam(lr=0.05, lr_decay=5e-7)

# Training model
EPOCHS = 10_000
for epoch in range(EPOCHS+1):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = activation_loss.forward(dense2.output, y)

    # Metrics
    accuracy = Accuracy(activation_loss.output, y)
    if epoch % 100 == 0:
        print(
            f"epoch: {epoch} acc: {accuracy:.3f} loss: {loss:.3f} lr: {optimizer.current_lr:.3f}")

    # Backpass
    activation_loss.backward(activation_loss.output, y)
    dense2.backward(activation_loss.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update params
    optimizer.pre_update_params()
    optimizer.update_params(dense2)
    optimizer.update_params(dense1)
    optimizer.post_update_params()

# Testing model
X_test, y_test = spiral_data(100, 3)

# Forward pass
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = activation_loss.forward(dense2.output, y_test)

# Metrics
accuracy = Accuracy(activation_loss.output, y_test)
print("\nTesting Metrics")
print(f"acc: {accuracy:.3f} loss: {loss:.3f}\n")
