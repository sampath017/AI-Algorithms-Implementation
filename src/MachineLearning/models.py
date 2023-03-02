import numpy as np


class LinearRegression:
    def __init__(self):
        self.name = "LinearRegression"
        self.X_true = np.nan
        self.y_true = np.nan
        self.m = 3.0
        self.b = 1.0
        self.output = np.nan

        self.dm = 0.0
        self.db = 0.0
        self.dvalue = 1.0

    def forward(self, X_true, y_true):
        self.X_true = X_true
        self.y_true = y_true

        self.output = self.m * self.X_true + self.b

    def backprop(self, dvalue):
        self.dvalue = dvalue
        n_samples = self.X_true.shape[0]

        self.dm = self.dvalue * \
            (2 / n_samples) * np.sum((self.output - self.y_true) * self.X_true)
        self.db = self.dvalue * (2 / n_samples) * \
            np.sum((self.output - self.y_true))
