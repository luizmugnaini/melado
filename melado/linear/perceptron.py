# Author: Luiz G. Mugnaini A.
import numpy as np
from numpy._typing import ArrayLike, NDArray


class Perceptron:
    def __init__(self, random_state: (int | None) = None):
        """Perceptron"""
        self.weights: NDArray | None = None
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike):
        X = np.asarray(X)
        y = np.asarray(y)

        # Assert the correctness of the arguments
        assert len(X.shape) == 2
        n_datapoints, dim_features = X.shape
        assert n_datapoints == len(y)

        # Preprocess `X`
        padding = np.repeat(1.0, n_datapoints)
        X = np.column_stack((X, padding))

        # Initialize weights with zero values to ensure that the model is wrong for all points in the first iteration
        self.weights = np.zeros(dim_features + 1)

        if isinstance(self.random_state, int):
            np.random.seed(self.random_state)

        epoch_indices = np.arange(n_datapoints)
        while True:
            if isinstance(self.random_state, int):
                np.random.shuffle(epoch_indices)

            n_misclassifications = 0
            for idx in epoch_indices:
                wrong_classification = y[idx] * np.dot(self.weights, X[idx]) <= 0
                if wrong_classification:
                    self.weights += y[idx] * X[idx]
                    n_misclassifications += 1

            if n_misclassifications == 0:
                break

    def predict(self, X: ArrayLike) -> NDArray:
        assert self.weights is not None
        X = np.asarray(X)
        padding = np.repeat(1.0, X.shape[0])
        X = np.column_stack((X, padding))

        return np.array([1 if np.dot(self.weights.T, x) >= 0 else -1 for x in X])

    def _weights(self):
        return self.weights
