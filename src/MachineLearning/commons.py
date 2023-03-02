import numpy as np

from .models import LinearRegression
from .losses import MeanSquaredError


class Model_LinearRegression_Loss_Mean_Squared_Error:
    def __init__(self):
        self.name = "LinearRegression"
        self.loss_function = MeanSquaredError()
        self.model = LinearRegression()
        self.dvalue = 1.0

    def forward(self, X_true, y_true):
        self.model.forward(X_true, y_true)
        self.output = self.model.output

        self.loss_function.calculate(y_true, self.output)

    def backprop(self, dvalue=1.0):
        self.dvalue = dvalue
        self.model.dm = self.dvalue * 2*np.mean(self.model.X_true * (self.model.X_true *
                                                                     self.model.m + self.model.b - self.model.y_true))

        self.model.db = self.dvalue * 2*np.mean(self.model.X_true * self.model.m +
                                                self.model.b - self.model.y_true)
