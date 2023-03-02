"""
This module contains the optimizers for various machine learning tasks.

classes:
    GradientDecent
"""

import numpy as np


class GradientDecent:
    """
    A class that calculates gradients for all samples at once.

    Attributes:
        lr (float): The learning rate.
        model (object): Model used to calculate gradients for (like linear regression model). 

    Methods:
        update_params(): Updates the model parametes by apply gradient decent.
    """

    def __init__(self, *, model=None, lr=0.1):
        self.lr = lr
        self.model = model

    def update_params(self):
        """
        Updates the model parameters by applying gradient decent method.

        Parameters:
            None

        Returns:
            None.
        """
        if self.model.name == "LinearRegression":
            self.model.m += -self.lr * self.model.dm
            self.model.b += -self.lr * self.model.db
        else:
            print("No known models are found.")
