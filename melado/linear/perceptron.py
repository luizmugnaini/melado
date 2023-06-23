"""Module containing the perceptron model."""
# Author: Luiz G. Mugnaini A.
import numpy as np
from numpy._typing import ArrayLike, NDArray


class Perceptron:
    """Perceptron model.

    Parameters
    ----------
    random_state : int | None
        Random state used for shuffling the datapoints for each epoch.
    max_iter : int
        Maximum number of iterations of the learning procedure.

    Attributes
    ----------
    weights : NDArray
        Model weights after the training stage.
    """

    def __init__(self, random_state: (int | None) = None, max_iter: int = 1000):
        self.random_state = random_state
        self.max_iter = max_iter
        self.is_fit = False

    def fit(self, X: ArrayLike, y: ArrayLike):
        """Fit training data to model.

        Parameters
        ----------
        X : ArrayLike, shape = (n_datapoints, n_features)
            Training set of datapoints.
        y : ArrayLike, shape = (n_datapoints,)
            True binary labeling of hte training datapoints `X`. We assume that
            the class labels are either `1` or `-1`.

        Returns
        -------
        self : Perceptron
            Model fitted to the training data.
        """
        X, y = np.asarray(X), np.asarray(y)
        n_datapoints, dim_features = X.shape
        assert n_datapoints == len(y)

        X = np.column_stack((X, np.ones(n_datapoints)))

        # Initialize weights with zero values to ensure that the model is wrong
        # for all points in the first iteration.
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

        self.is_fit = True
        return self

    def predict(self, X: ArrayLike) -> NDArray:
        """Returns the predicted classification.

        Parameters
        ----------
        X : ArrayLike
            Datapoints to be classified by the model.

        Returns
        -------
        prediction : NDArray
            Predicted labels for the given datapoints.
        """
        assert self.is_fit, "The model should be fitted before being used for predictions."
        X = np.asarray(X)
        padding = np.repeat(1.0, X.shape[0])
        X = np.column_stack((X, padding))

        return np.array([1 if np.dot(self.weights.T, x) >= 0 else -1 for x in X])
