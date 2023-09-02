"""Module containing the perceptron model."""
# Author: Luiz G. Mugnaini A.
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Optional, Self


# TODO:
# * Implement the pocket algorithm for the perceptron (can be found in "Learning from Data").
# * See page 97 (paragraph just above exercise 3.9) for a way to speed up the fitting of the
#   perceptron using the linear regression as a hot start for the weights (instead of using
#   a random vector).
class Perceptron:
    """Perceptron model.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations of the learning procedure. Default is 1000.
    random_state : Optional[int]
        Random state used for shuffling the datapoints for each epoch. Default is `None`

    Attributes
    ----------
    weights : NDArray
        Model weights. Will only be created after the training stage.
    """

    __slots__ = ("random_state", "max_iter", "is_fit", "weights")

    def __init__(self, max_iter: int = 1000, random_state: Optional[int] = None) -> None:
        self.max_iter = max_iter
        self.random_state = random_state
        self.is_fit = False

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit training data to model.

        Parameters
        ----------
        X : ArrayLike, with shape `(n_datapoints, n_features)`
            Training set of datapoints.
        y : ArrayLike, with shape `(n_datapoints,)`
            True binary labeling of the training datapoints `X`. We assume that
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

        epoch_idx = np.arange(n_datapoints)
        while True:
            if isinstance(self.random_state, int):
                np.random.shuffle(epoch_idx)

            n_misclassifications = 0
            for idx in epoch_idx:
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
        assert (
            self.weights is not None
        ), "In order to obtain a prediction, the weights must be computed previously."

        X = np.asarray(X)
        padding = np.repeat(1.0, len(X))
        X = np.column_stack((X, padding))

        return np.fromiter((1 if np.dot(self.weights.T, x) >= 0 else -1 for x in X), float, len(X))
