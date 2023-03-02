"""
This module contains the classes for models and losses which works by computing gradients together.

classes:
    Model_LinearRegression_Loss_Mean_Squared_Error
"""


import numpy as np

from .models import LinearRegression
from .losses import MeanSquaredError


class Model_LinearRegression_Loss_Mean_Squared_Error:
    """
    A class that performs sequentional forward pass for model and loss. And caluculates combined gradients for both linear regression and mean squared error loss.

    Attributes:
        loss_function (object): The loss function to calculate the loss of the model used.
        model (object): The model used to train and test. 
        dvalue (float): The gradient recived from the previous layer during backpropagation.

    Methods:
        forward(X_true, y_true): performs forward pass through model and calculates loss.
        backprop(dvalue=1.0): Calcuates gradients by apply backpropagation from the previous layer gradient.

    Notes:
        Here `layer` means a function in the process of training like a model, loss.
    """

    def __init__(self):
        self.loss_function = MeanSquaredError()
        self.model = LinearRegression()
        self.dvalue = 1.0

    def forward(self, X_true, y_true):
        """
        Forward passes X_true through the model of f(x) = x*m + b and calculates loss.

        Parameters:
            X_true (ndarray): Features used to train the model (n_samples, ). 
            y_true (ndarray): Labels used to train the model (n_samples, ).

        Returns:
            None.
        """
        self.model.forward(X_true, y_true)
        self.output = self.model.output

        self.loss_function.calculate(y_true, self.output)

    def backprop(self, dvalue=1.0):
        """
        Calculates gradients by apply backpropagation from the previous layer gradient for each parameter of the model.

        Parameters:
            dvalue (float): The gradient recived from the previous layer during backpropagation.

        Returns:
            None.
        """
        self.dvalue = dvalue
        self.model.dm = self.dvalue * 2*np.mean(self.model.X_true * (self.model.X_true *
                                                                     self.model.m + self.model.b - self.model.y_true))

        self.model.db = self.dvalue * 2*np.mean(self.model.X_true * self.model.m +
                                                self.model.b - self.model.y_true)
