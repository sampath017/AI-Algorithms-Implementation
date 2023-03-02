import numpy as np


class GradientDecent:
    def __init__(self, *, model=None, lr=0.1):
        self.lr = lr
        self.model = model

    def update_params(self):
        if self.model.name == "LinearRegression":
            self.model.m += -self.lr * self.model.dm
            self.model.b += -self.lr * self.model.db
        else:
            print("No known models are found.")
