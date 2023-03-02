import numpy as np


class RootMeanSquaredError:

    def __init__(self):
        self.y_true = np.nan
        self.y_pred = np.nan
        self.output = None
        self.dinput = 0.0
        self.dvalue = 1.0

    def calculate(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

        self.output = np.sqrt(np.mean(np.square(self.y_true - self.y_pred)))

    def backprop(self, dvalue=1.0):
        self.dvalue = dvalue
        self.dinput = self.dvalue / (2 * self.output)


class MeanSquaredError:
    def __init__(self):
        self.y_true = np.nan
        self.y_pred = np.nan
        self.output = None
        self.dinput = 0.0
        self.dvalue = 1.0

    def calculate(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

        self.output = np.mean(np.square(self.y_true - self.y_pred))
