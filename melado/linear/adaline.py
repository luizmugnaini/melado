"""Module containing the adaline model."""
# Author: Luiz G. Mugnaini A.
import numpy as np
from numpy._typing import ArrayLike, NDArray


# TODO: Write tests for the adaline model.
class Adaline:
    """Adaptative Linear Neuron Classifier (adaline).

    Parameters
    ----------
    learning_rate : float
        Learning rate is a value between `0.0` and `1.0` that informs how
        significant will be the change in the weights of the model for each
        epoch with respect to the gradient descent method. Defaults to `0.01`.
    max_iter : int
        Maximum number of iterations (epochs) executed by the learning
        algorithm. Defaults to `1000`.
    tol : float | None
        Allowed tolerance for the mean squared error. Setting to `None` has the
        same behaviour as having `0.0` tolerance. Defaults to `0.001`.
    random_state : int | None
        Random state used for initializing the weights of the model. Defaults to `None`.

    Attributes
    ----------
    weights : NDArray
        Model weights. The value `weights[0]` corresponds to the bias parameter
        of the model, while each element of `weights[1:]` corresponds to the
        weight associated to each feature.
    n_features : int
        Number of features seen in the training stage.
    epoch_mse : list
        List containing the mean squared error with respect to each epoch in
        the training stage.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: (float | None) = 0.001,
        random_state: (int | None) = None,
    ):
        assert (
            0.0 <= learning_rate <= 1.0
        ), f"The learning rate should be between 0.0 and 1.0, instead got {learning_rate}."
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.is_fit = False

    def fit(self, X: ArrayLike, y: ArrayLike):
        """Fit training data to the model.

        Parameters
        ----------
        X : ArrayLike, shape = (n_datapoints, n_features)
            Datapoints used for training the model.
        y : ArrayLike, shape = (n_datapoints,)
            True binary labels for `X`, which we assume are either `1` or `-1`.

        Returns
        -------
        self : Adaline
            The fitted model.
        """
        X, y = np.asarray(X), np.asarray(y)
        assert len(X) == len(y), f"Lengths of X ({len(X)}) and y ({len(y)}) don't match."
        n_datapoints, self.n_features = X.shape
        if isinstance(self.random_state, int):
            np.random.seed(self.random_state)

        # We start the learning algorithm with small random weights, and for
        # each iteration we apply the gradient descent method to the weights.
        #
        # Two stopping conditions are possible: either we go through all
        # iterations or we stop short after we meet the required `err_bound`.
        self.epoch_mse = []
        self.weights = np.random.normal(scale=0.01, size=self.n_features + 1)
        self.is_fit = True
        for _ in range(self.max_iter):
            y_hat = self.predict(X)
            error = y - y_hat

            # We compute separately the gradient for the weights
            # (`self.weights[1:]`) and the bias term (`self.weights[0]`), then
            # update our coefficients via gradient descent.
            gradient = (2.0 * np.dot(X.T, error) / n_datapoints, error.sum())
            self.weights[1:] += self.learning_rate * gradient[0]
            self.weights[0] += self.learning_rate * gradient[1]

            self.epoch_mse.append((error**2).sum() / n_datapoints)
            if (self.tol is not None) and self.epoch_mse[-1] < self.tol:
                break
        return self

    def predict(self, X: ArrayLike) -> NDArray:
        """Classify a given array of datapoints.

        Parameters
        ----------
        X : ArrayLike, shape = (n_datapoints, n_features)
            Datapoints to be classified by the model. The number of features
            should be the same as the amount seen in the training stage.

        Returns
        -------
        prediction : NDArray
            The array containing the predicted classification of each datapoint
            according the following threshold: if the aggregation function for
            the point is non-negative, we classify it as `1`, otherwise the
            point is classified as `-1`.
        """
        assert self.is_fit, "The model should be fitted to a training set before any predictions."

        X = np.asarray(X)
        assert (
            X.shape[1] == self.n_features
        ), f"Expected {self.n_features} features from X but got {X.shape[0]}"

        return np.where(self._aggregate(X) >= 0.0, 1, -1)

    def _aggregate(self, X: NDArray) -> NDArray:
        """Linear aggregation function.

        For each datapoint in `X`, computes the linear combination of the point
        with the model weights and bias parameter.
        """
        return np.dot(X, self.weights[1:]) + self.weights[0]
