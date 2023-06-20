"""Module containing the linear regression model."""
# Author: Luiz G. Mugnaini A.
from typing import Callable
from collections.abc import Set
import numpy as np
from numpy._typing import NDArray, ArrayLike


# NOTE: The use of base functions still needs extensive testing. Maybe the
# general method should be replaced with a weighted linear regression, which I
# think may be more useful at this point and easier to test.
class LinearRegression:
    """Linear regression model.

    Parameters
    ----------
    None

    Attributes
    ----------
    weights : ndarray
        Weights of the model after training.
    n_features : int
        Number of features seen in the training stage.
    """
    def __init__(self):
        self.is_fit = False

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        base_fns: (Set[Callable[[NDArray], float]] | None) = None,
    ):
        """Fit the model to the training set.

        Parameters
        ----------
        X : array_like, with shape `(n_datapoints, n_features)`
            Training set of datapoints.
        y : array_like, with shape `(n_datapoints, n_target)`
            True target value of the datapoints `X`. Each value associated to a
            datapoint can be a scalar or an array with dimension `n_target`.
        base_fns : Set[Callable[[NDArray], float]] | None
            Set of base functions associated with each feature.

        Returns
        -------
        self : LinearRegression
            Fitted model.
        """
        self.base_fns = base_fns
        X, y = self._pre_process(X, y)
        X_pseudo_inverse = np.linalg.pinv(X)
        self.weights = X_pseudo_inverse @ y
        self.is_fit = True
        return self

    def predict(self, X: ArrayLike) -> NDArray:
        """Returns the predicted values associated with a set of datapoints.

        Parameters
        ----------
        X : array_like, shape = (n_datapoints, n_features)
            Datapoints to be evaluated by the model.

        Returns
        -------
        prediction : ndarray
            Predicted values associated with `X`.
        """
        assert self.is_fit, "The model should be fitted before it can be used for predictions."
        X = np.asarray(X)
        assert X.shape[1] == self.n_features, f"Expected datapoints to have {self.n_features} but got {X.shape[1]}"
        X = np.column_stack((np.ones(X.shape[0]), X))

        return X @ self.weights

    # TODO: Implement the update method using the Sherman-Morrison formula.
    def update(self, X: ArrayLike, y: ArrayLike, base_fns: Set[Callable[[NDArray], float]]):
        """Update the model with a new training set.

        Parameters
        ----------
        X : array_like, shape = (n_datapoints, n_features)
            New training set of datapoints.
        y : array_like, shape = (n_datapoints,)
            True target value of the datapoints `X`.
        base_fns : Set[Callable[[NDArray], float]] | None
            Set of base functions associated with each datapoint.

        Returns
        -------
        self : LinearRegression
            Updated model.
        """
        assert self.is_fit, "The model needs to be fitted before being updated."
        pass

    def _pre_process(self, X: ArrayLike, y: ArrayLike) -> tuple[NDArray, NDArray]:
        r"""Pre-processing of the dataset.

        Returns
        -------
        design_matrix : ndarray, with shape `(n_datapoints, n_features)`
            Let :math:`(\phi_j)` be the collection of base functions. The
            design matrix :math:`\Phi = [\Phi_{n j}]` is defined as
            :math:`\Phi_{n j} = \phi_j(x_j)`.
        """
        X, y = np.asarray(X), np.asarray(y)
        n_datapoints, self.n_features = X.shape
        assert n_datapoints == len(y), f"The lengths of X ({n_datapoints}) and y ({len(y)}) don't match."

        # Given a base function, create the design matrix from `X`
        if self.base_fns is not None:
            assert len(self.base_fns) == n_datapoints, f"Expected base function to have shape ({n_datapoints},) but got {len(self.base_fns)}"
            for n in range(n_datapoints):
                for j, f_j in enumerate(self.base_fns):
                    X[n, j] = f_j(X[n])

        # The padding of ones is applied to the matrix `X` in order to smoothly deal
        # with the weights when taking products of the form `X @ weights`. The padding
        # is applied after the calculating the base function on `X` since we don't want
        # to apply the base function to the bias parameter of the model but solely on
        # the features of the given dataset.
        X = np.column_stack((np.ones(n_datapoints), X))
        self.weights = np.zeros(n_datapoints + 1)

        return X, y
