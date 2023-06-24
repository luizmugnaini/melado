"""Module containing the linear regression model."""
# Author: Luiz G. Mugnaini A.
import numpy as np
from numpy._typing import NDArray, ArrayLike


class LinearRegression:
    r"""Weighted linear regression model.

    Suppose we want to model an :math:`n`-multivariate variable :math:`y` which is given by a
    deterministic function and additive Gaussian noise:

    .. math::

        y = X \beta + \varepsilon

    That is, the noise :math:`\varepsilon` is an :math:`n`-multivariate variable with normal
    distribution, zero mean and non-constant variance :math:`\Sigma`, where:

    .. math::

        \Sigma =
            \begin{bmatrix}
                \sigma_1 &0 &\dots &0 \\
                        0 &\sigma_2 &\dots &0
                            \vdots &\ddots &\vdots \\
                                0 &0 &\dots &\sigma_n
                                    \end{bmatrix}

    Define a weighting matrix :math:`W` to be the diagonal matrix of the precision of
    :math:`\varepsilon` (recall that the precision is defined as the reciprocal of the variance):

    .. math::
        W =
            \begin{bmatrix}
                1/\sigma_1^2 &0 &\dots &0 \\
                        0 &1/\sigma_2^2 &\dots &0
                            \vdots &\ddots &\vdots \\
                                0 &0 &\dots &1/\sigma_n^2
                                    \end{bmatrix}

    Our goal is to find the best linear unbiased estimator :math:`\widehat{\beta}_{\text{WLS}}`
    via least squares and maximum likelihood. It can be shown that:

    .. math::
        \widehat{\beta}_{\text{WLS}} =
            (X^\top W X)^{-1} X^\top W y


    defining a new :math:`n \times m` matrix :math:`D` whose entries are given by
    :math:`D_{ij} = W_i (X_{\text{sample}})_{ij}`.

    The matrix :math:`D` is called the design matrix. The weighted linear regression
    algorithm can then be used to solve the linear system :math:`D L = y_{\text{sample}}`.

    References
    ----------
    .. [1] https://online.stat.psu.edu/stat501/lesson/13

    Parameters
    ----------
    fit_intercept : bool
        Toggle intercept fitting, default is `True`


    Attributes
    ----------
    coef : NDArray
        Coefficients (weights) of the model after training.
    n_features : int
        Number of features seen in the training stage.
    inv_cov : NDArray | None
        Inverse of the covariance matrix associated to the training datapoints. This is only computed
        if weights are provided.
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.is_fit = False

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        W: (ArrayLike | None) = None,
    ):
        r"""Fit the model to the training set.

        We now explain how to fit the weighted linear regression model. Assume we get a sample
        of :math:`n` i.i.d. points :math:`x_1, \dots, x_n \in \mathbf{R}^m` with associated true
        values :math:`y_1 := y(x_1), \dots, y_n := y(x_n) \in \mathbf{R}^k`. Define matrices

        .. math::

            X_{\text{sample}}
                := [x_1, x_2, \dots, x_n]^\top \in \mathbf{R}^n \times \mathbf{R}^m
                    \quad \text{and} \quad
                        y_{\text{sample}}
                            = [y_1, y_2, \dots y_n]^\top \in \mathbf{R}^n \times \mathbf{R}^k

        Moreover, assume we know some previous information that allows us to decide
        that a certain datapoint of the sample has a greater relevance. With this
        we can define a weighting vector :math:`W \in \mathbf{R}^n` where each coordinate
        :math:`W_j` corresponds to the weight associated with the sample :math:`x_j`. Ideally,
        we should have :math:`W_j = 1/\sigma_j^2`, where :math:`\sigma_j^2` is the variance
        associted with :math:`x_j`.

        Note
        ----

        Notice that if we set :math:`W_j = 1` for each index :math:`j` then the weighted linear
        regression reduces to the ordinary linear regression.


        Parameters
        ----------
        X : ArrayLike, with shape `(n_datapoints, n_features)`
            Training set of datapoints.
        y : ArrayLike, with shape `(n_datapoints, n_target)`
            True target value of the datapoints `X`. Each value associated to a
            datapoint can be a scalar or an array with dimension `n_target`.
        W : ArrayLike | None, with shape `(n_datapoints,)`
            Weights associated with each datapoint of `X`. When `W` is set to `None`, the algorithm
            performs the ordinary linear regression on `X` and `y`.

        Returns
        -------
        self : LinearRegression
            Fitted model.
        """
        X, y = self._pre_process(X, y, W)
        self.coef = self.inv_cov @ X.T @ y
        print(
            "X = {}, y = {}, inv_cov = {}, coef = {}".format(
                X.shape, y.shape, self.inv_cov.shape, self.coef.shape
            )
        )
        self.is_fit = True
        return self

    def predict(self, X: ArrayLike) -> NDArray:
        """Returns the predicted values associated with a set of datapoints.

        Parameters
        ----------
        X : ArrayLike, shape = `(n_datapoints, n_features)`
            Datapoints to be evaluated by the model.

        Returns
        -------
        prediction : NDArray
            Predicted values associated with `X`.
        """
        assert self.is_fit, "The model should be fitted before it can be used for predictions."

        X = np.asarray(X)
        features_error = f"Expected datapoints to have {self.n_features} but got {X.shape[1]}"
        assert X.shape[1] == self.n_features, features_error

        if self.fit_intercept:
            X = np.column_stack((np.ones(X.shape[0]), X))

        return X @ self.coef

    # TODO: Implement the update method using the Sherman-Morrison formula.
    def update(self, X: ArrayLike, y: ArrayLike, W: ArrayLike):
        """Update the model with a new training set.

        Parameters
        ----------
        X : ArrayLike, with shape `(n_datapoints, n_features)`
            New training set of datapoints.
        y : ArrayLike, with shape `(n_datapoints,)`
            True target value of the datapoints `X`.
        W : ArrayLike, with shape `(n_datapoints,)`
            Weights associated with each datapoint of `X`

        Returns
        -------
        self : LinearRegression
            Updated model.
        """
        # assert self.is_fit, "The model needs to be fitted before being updated."
        raise NotImplementedError("Yet to be implemented")

    def _pre_process(
        self,
        X: ArrayLike,
        y: ArrayLike,
        W: (ArrayLike | None) = None,
    ) -> tuple[NDArray, NDArray]:
        r"""Pre-processing of the dataset.

        Parameters
        ----------
        X : ArrayLike, with shape `(n_datapoints, n_features)`
            Datapoints to be preprocessed.
        y : ArrayLike, with shape `(n_datapoints,)`
            True values associated to `X`.
        W : ArrayLike, with shape `(n_datapoints,)`
            Weights associated with each datapoint of `X`

        Returns
        -------
        (design_matrix, y) : tuple[NDArray, NDArray]
        """
        X, y = np.asarray(X), np.asarray(y)
        print(X.shape)

        assert X.ndim <= 2, f"Expected X to have at most dimension 2, but got {X.ndim}"
        n_datapoints, self.n_features = X.shape if len(X.shape) == 2 else (len(X), 1)

        len_error = f"The lengths of X ({n_datapoints}) and y ({len(y)}) don't match."
        assert n_datapoints == len(y), len_error

        if W is not None:
            W = np.atleast_1d(W).squeeze()
            weights_error = f"Expected weights to have shape ({n_datapoints},), but got {W.shape}"
            assert W.shape == (n_datapoints,), weights_error

            # Rescaling of `X` and `y`.
            W_sqrt = np.diag((_W_sqrt := np.sqrt(W)))
            X, y = W_sqrt @ X, W_sqrt @ y

            if self.fit_intercept:
                # Since the intercept column should also be rescaled, we stack the square root
                # of the weights instead of a vector of ones.
                X = np.column_stack((_W_sqrt, X))
        elif self.fit_intercept:
            X = np.column_stack((np.ones(n_datapoints), X))

        self.inv_cov = np.linalg.pinv(X.T @ X)
        return X, y
