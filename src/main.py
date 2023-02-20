import nnfs
from nnfs.datasets import spiral_data, sine_data

import numpy as np

from network.layers import Dense, Dropout
from network.activations import ReLU, Sigmoid, Linear, Softmax
from network.commons import ActivationSoftmaxLossCategoricalCrossentropy
from network.optimizers import Adam
from network.losses import BinaryCrossentropy, MeanSquaredError, CategoricalCrossentropy
from network.models import Model
from network.accuracys import Regression, Categorical
from network.gen_data import create_data_mnist

nnfs.init()

X_train, y_train, X_test, y_test = create_data_mnist("data")

# Shuffle
keys = np.array(range(X_train.shape[0]))
np.random.shuffle(keys)
X_train = X_train[keys]
y_train = y_train[keys]

# Scale and reshape
X_train = (X_train.reshape(
    X_train.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(
    X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

# model.add(Dense(X_train.shape[1], 128))
# model.add(ReLU())
# model.add(Dense(128, 128))
# model.add(ReLU())
# model.add(Dense(128, 10))
# model.add(Softmax())

# model.compile(
#     loss=CategoricalCrossentropy(),
#     optimizer=Adam(lr_decay=1e-3),
#     accuracy=Categorical()
# )

# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, print_every=100,
#           batch_size=128)

model = Model.load("fashion_mnist.model")

confidences = model.predict(X_test[:5])
predictions = model.output_layer_activation.predictions(confidences)

print(predictions)
print(y_test[:5])
