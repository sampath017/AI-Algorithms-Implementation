"""
This module contains the models for various machine learning tasks.

classes:
    LinearRegression
"""

import numpy as np


class LinearRegression:
    """
    A class that performs linear regression.

    Attributes:
        name (str) : Name of the model
        X_true (ndarray): Features used to train the model. 
        y_true (ndarray): Labels used to train the model.
        m (float): Slope of the linear regression line
        b (float): y-intercept of the linear regression line.
        output (ndarray): The output of the model after a forward pass.

    Methods:
        forward(X_true, y_true): Forward passes X_true through the model.

    Notes:
        This implementation is limited to linear regression with a single feature.
    """

    def __init__(self):
        self.name = "LinearRegression"
        self.X_true = np.nan
        self.y_true = np.nan
        self.m = 3.0
        self.b = 1.0
        self.output = np.nan

    def forward(self, X_true, y_true):
        """
        Forward passes X_true through the model of f(x) = x*m + b.

        Parameters:
            X_true (ndarray): Features used to train the model (n_samples, ). 
            y_true (ndarray): Labels used to train the model (n_samples, ).

        Returns:
            The models output after the forward pass through f(x).
        """
        self.X_true = X_true
        self.y_true = y_true

        self.output = self.m * self.X_true + self.b
