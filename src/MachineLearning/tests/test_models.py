import numpy as np

from ..models import LinearRegression


class TestLinearRegression:
    def test_forward(self):
        X_true = np.array([1, 2, 3, 4])
        y_true = np.array([2, 4, 6, 8])
        model = LinearRegression(X_true, y_true)
        model.m = 2
        model.b = 0
        expected_output = np.array([2, 4, 6, 8])

        np.testing.assert_allclose(model.forward(), expected_output)

    def test_zero_slope(self):
        X_true = np.array([1, 2, 3, 4])
        y_true = np.array([2, 4, 6, 8])
        model = LinearRegression(X_true y_true)
        model.m = 0
        model.b = 1
        expected_output = np.array([1, 1, 1, 1])

        np.testing.assert_allclose(model.forward(), expected_output)

    def test_zero_intercept(self):
        X_true = np.array([2, 4, 6, 8])
        y_true = np.array([4, 8, 12, 16])
        model = LinearRegression(X_true, y_true)
        model.m = 2
        model.b = 0
        expected_output = np.array([4, 8, 12, 16])

        np.testing.assert_allclose(model.forward(), expected_output)
