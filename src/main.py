import nnfs
from nnfs.datasets import spiral_data, sine_data

import numpy as np
import matplotlib.pyplot as plt

from network.layers import Dense, Dropout
from network.activations import ReLU, Sigmoid, Linear, Softmax
from network.commons import ActivationSoftmaxLossCategoricalCrossentropy
from network.optimizers import Adam
from network.losses import BinaryCrossentropy, MeanSquaredError, CategoricalCrossentropy
from network.models import Model
from network.accuracys import Regression, Categorical

nnfs.init()

X_train, y_train = spiral_data(samples=1000, classes=3)

model = Model()

model.add(Dense(2, 512, l2_weight_regularizer=5e-4, l2_bias_regularizer=5e-4))
model.add(ReLU())
model.add(Dropout(0.1))
model.add(Dense(512, 3))
model.add(Softmax())

model.compile(
    loss=CategoricalCrossentropy(),
    optimizer=Adam(lr=0.05, lr_decay=5e-5),
    accuracy=Categorical()
)

model.fit(X_train, y_train, epochs=10_000, print_every=100)

# Evaluate model
X_test, y_test = spiral_data(samples=100, classes=3)

model.evaluate(X_test, y_test)
