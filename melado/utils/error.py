import numpy as np
from numpy._typing import NDArray


class Error:
    @classmethod
    def mse(cls, predict: NDArray, actual: NDArray) -> float:
        """Mean Squared Error"""
        assert len(predict) == len(actual)
        m = len(predict)
        error = float(2 / m * np.sum((predict - actual) ** 2))
        return error
