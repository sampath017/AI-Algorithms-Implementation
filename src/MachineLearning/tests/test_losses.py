import numpy as np
from sklearn.metrics import mean_squared_error

from ..losses import RootMeanSquaredError


class TestRootMeanSquaredError:
    def test_calculate_small_values(self):

        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([2, 3, 4, 5])

        RootMeanSquaredError = RootMeanSquaredError(y_true, y_pred)

        assert np.isclose(RootMeanSquaredError.calculate(), 1.0)

    def test_calculate_low_values(self):
        y_true = np.random.rand(100000) * 1e-10
        y_pred = np.random.rand(100000) * 1e-10

        RootMeanSquaredError = RootMeanSquaredError(y_true, y_pred)
        RootMeanSquaredError_score = RootMeanSquaredError.calculate()

        mse = mean_squared_error(y_true, y_pred)
        RootMeanSquaredError_manual = np.sqrt(mse)

        assert np.isclose(RootMeanSquaredError_score, RootMeanSquaredError_manual)

    def test_calculate_high_values(self):
        y_true = np.random.rand(100000) * 1e12
        y_pred = np.random.rand(100000) * 1e12

        RootMeanSquaredError = RootMeanSquaredError(y_true, y_pred)
        RootMeanSquaredError_score = RootMeanSquaredError.calculate()

        mse = mean_squared_error(y_true, y_pred)
        RootMeanSquaredError_manual = np.sqrt(mse)

        assert np.isclose(RootMeanSquaredError_score, RootMeanSquaredError_manual)
