import json
import nnfs
from nnfs.datasets import spiral_data

from network.layers import Dense, Dropout
from network.activations import ReLU
from network.commons import ActivationSoftmaxLossCategoricalCrossentropy
from network.optimizers import Adam
from network.metrics import Accuracy

nnfs.init()

X, y = spiral_data(1000, 3)
X_test, y_test = spiral_data(100, 3)

dense1 = Dense(2, 512, l2_weight_regularizer=5e-4, l2_bias_regularizer=5e-4)
activation1 = ReLU()
dropout1 = Dropout(0.1)
dense2 = Dense(512, 3)
activation_loss = ActivationSoftmaxLossCategoricalCrossentropy()
optimizer = Adam(lr=0.05, lr_decay=5e-5)

# Training model
EPOCHS = 10_000
training_history = {'epochs': EPOCHS, 'accuracy': [], 'loss': [], 'lr': []}

for epoch in range(EPOCHS+1):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)

    data_loss = activation_loss.forward(dense2.output, y)
    regularization_loss = activation_loss.loss.regularization_loss(
        dense1) + activation_loss.loss.regularization_loss(dense2)
    loss = data_loss + regularization_loss

    # Metrics
    accuracy = Accuracy(activation_loss.output, y)

    training_history['accuracy'].append(float(accuracy))
    training_history['loss'].append(float(loss))
    training_history['lr'].append(float(optimizer.current_lr))
    if epoch % 100 == 0:
        print(
            f"epoch: {epoch} acc: {accuracy:.3f} (data_loss: {data_loss:.3f} regularization_loss: {regularization_loss:.3f}) loss: {loss:.3f}  lr: {optimizer.current_lr:.3f}")

    # Backpass
    activation_loss.backward(activation_loss.output, y)
    dense2.backward(activation_loss.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    # Update params
    optimizer.pre_update_params()
    optimizer.update_params(dense2)
    optimizer.update_params(dense1)
    optimizer.post_update_params()

# Save training history to JSON file
with open('training_history.json', 'w') as f:
    json.dump(training_history, f)

# Testing model
# Forward pass
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = activation_loss.forward(dense2.output, y_test)

# Metrics
accuracy = Accuracy(activation_loss.output, y_test)
print("\nTesting Metrics")
print(f"acc: {accuracy:.3f} loss: {loss:.3f}\n")

# Save testing metrics to JSON file
testing_metrics = {'accuracy': float(accuracy), 'loss': float(loss)}
with open('testing_metrics.json', 'w') as f:
    json.dump(testing_metrics, f)
