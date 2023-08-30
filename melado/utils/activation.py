import numpy as np
from enum import Enum


class Logistic:
    """Logistic (sigmoid) activation function."""

    @staticmethod
    def __call__(x: float) -> float:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(x: float) -> float:
        return np.exp(x) / (1 + np.exp(x)) ** 2

    @staticmethod
    def derivative2(x: float) -> float:
        return 2 * np.exp(-2 * x) / (1 + np.exp(-x)) ** 3 - np.exp(-x) / (1 + np.exp(-x)) ** 2


class Heaviside:
    """Heaviside activation function.

    This is the function used by the perceptron model, given an input `x`, if `x` is positive
    returns 1 else returns 0.
    """

    @staticmethod
    def __call__(x: float) -> float:
        return 1.0 if x > 0 else 0.0


class ReLu:
    """Rectifier Linear Unit activation function.

    Given an input `x` returns `max(0, x)`.
    """

    @staticmethod
    def __call__(x: float) -> float:
        return max(0.0, x)


class Activation(Enum):
    """Enumeration containing all implemented activation functions.

    Options
    -------
    logistic: Logistic
       Logistic (sigmoid) activation function.
    heaviside : Heaviside
        Heaviside activation function.
    """

    logistic = Logistic()
    heaviside = Heaviside()
