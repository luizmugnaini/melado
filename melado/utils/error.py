"""Module containing different error measures."""
import numpy as np
from numpy._typing import NDArray


class Error:
    @classmethod
    def mse(cls, predict: NDArray, actual: NDArray) -> float:
        """Mean Squared Error

        Parameters
        ----------
        predict : NDArray
            Values to be matched with `actual`.
        actual : NDArray
            True values.

        Returns
        -------
        error : float
            Mean squared error.
        """
        assert len(predict) == len(actual)
        m = len(predict)
        error = float(2 / m * np.sum((predict - actual) ** 2))
        return error
