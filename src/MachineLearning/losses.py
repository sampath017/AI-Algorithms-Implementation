"""
This module contains the loss functions for various machine learning tasks.

classes:
    MeanSquaredError
"""

import numpy as np


class MeanSquaredError:
    """
    A class that performs linear regression.

    Attributes:
        y_true (ndarray): True labels.
        y_pred (ndarray): Prediction labels. 
        output (ndarray): The loss of the model.

    Methods:
        calculate(y_true, y_pred): calculates the loss of the model.

    Notes:
        This implementation is supposed to be used with commens.py model.
    """

    def __init__(self):
        self.y_true = np.nan
        self.y_pred = np.nan
        self.output = None

    def calculate(self, y_true, y_pred):
        """
        Calculates the loss of the model by applying mean squred error function.

        Parameters:
            y_true (ndarray): True labels (n_samples, ). 
            y_pred (ndarray): Prediction labels (n_samples, ).

        Returns:
            None.
        """
        self.y_true = y_true
        self.y_pred = y_pred

        self.output = np.mean(np.square(self.y_true - self.y_pred))
