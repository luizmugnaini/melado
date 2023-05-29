# Author: Luiz G. Mugnaini A.
import numpy as np
from numpy._typing import NDArray, ArrayLike
from melado.utils.linear_sys import Qr


class LinearRegression:
    def __init__(self):
        """Linear regression model"""
        self.coefficients: NDArray | None = None
        self.training_set_size: int | None = None

    def _pre_process(self, X: ArrayLike, y: ArrayLike):
        X, y = np.asarray(X), np.asarray(y)
        sample_size = len(X)
        if sample_size != len(y):
            raise ValueError(
                f"The lengths of X ({len(X)}) and y ({len(y)}) don't match."
            )

        padding = np.repeat(1.0, sample_size)
        X = np.column_stack((padding, X))
        self.coefficients = np.zeros(sample_size + 1)
        return X, y

    def fit(self, X: ArrayLike, y: ArrayLike):
        X, y = self._pre_process(X, y)
        A = np.matmul(X.T, X)
        try:
            # Case where `A` is invertible.
            X_pseudo_inverse = np.matmul(np.linalg.inv(A), X.T)
            self.coefficients = np.matmul(X_pseudo_inverse, y)
        except:
            # In the case where `A` is singular we use the least-norm solution.
            self.coefficients = Qr.solve(A, np.matmul(X.T, y))

    def predict(self, X: ArrayLike):
        assert self.coefficients is not None
        # Pre-process data.
        X = np.asarray(X)
        padding = np.repeat(1.0, len(X))
        X = np.column_stack((padding, X))

        prediction = np.array([np.dot(self.coefficients.T, x) for x in X])
        return prediction
