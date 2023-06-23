"""Module containing the linear regression model."""
# Author: Luiz G. Mugnaini A.
import numpy as np
from numpy._typing import NDArray, ArrayLike


class LinearRegression:
    r"""Weighted Linear regression model.

    Let :math:`y: \mathbf{R}^m \to \mathbf{R}^k` be an :math:`\mathbf{R}`-linear map
    which we want to find weights :math:`\ell_{i j} \in \mathbf{R}` such that

    .. math::

        y(v) = L v =
            \begin{bmatrix}
                \ell_{11} &\dots &\ell_{1 m} \\
                    \vdots &\ddots &\vdots \\
                        \ell_{k 1} &\dots &\ell_{k m}
                            \end{bmatrix}
                                \begin{bmatrix}
                                    v_1 \\ \vdots \\ v_m
                                        \end{bmatrix}

    In order to do that, assume get a sample of :math:`n` i.i.d. points
    :math:`x_1, \dots, x_n \in \mathbf{R}^m` with associated values
    :math:`y_1 := y(x_1), \dots, y_n := y(x_n) \in \mathbf{R}^k`. Define matrices

    .. math::

        X_{\text{sample}}
            := [x_1, x_2, \dots, x_n]^\top \in \mathbf{R}^n \times \mathbf{R}^m
                \quad \text{and} \quad
                    y_{\text{sample}}
                        = [y_1, y_2, \dots y_n]^\top \in \mathbf{R}^n \times \mathbf{R}^k

    Moreover, assume we know some previous information that allows us to decide
    that a certain datapoint of the sample has a greater relevance, or that a
    feature is more important than others. With this we can define a weighting matrix

    .. math::

        W \in \mathbf{R}^s \times \mathbf{R}^t

    where, depending on the information we have, admits

    .. math::

        (s, t) \in \{(n, 1), (m, 1) (n, m)\}.

    Each case can be described as follows:

    1. If :math:`(s, t) = (n, 1)` then :math:`W` can be used to weight each of the
       datapoints in :math:`X_{\text{sample}}` differently, defining a new
       :math:`n \times m` matrix :math:`D` whose entries are given by
       :math:`D_{ij} = W_i (X_{\text{sample}})_{ij}`.

    2. If :math:`(s, t) = (m, 1)` then :math:`W` can be used to weight each of the
       features (columns) in :math:`X_{\text{sample}}` differently, defining a new
       :math:`n \times m` matrix :math:`D` whose entries are given by
       :math:`D_{ij} = W_j (X_{\text{sample}})_{ij}`.

    3. If :math:`(s, t) = (n, m)` then :math:`W` can be used to weight each of the
       features (columns) in :math:`X` differently, defining a new :math:`n \times m`
       matrix :math:`D` whose entries are given by
       :math:`D_{ij} = W_{ij} (X_{\text{sample}})_{ij}`.

    The matrix :math:`D` is called the design matrix. The weighted linear regression
    algorithm can then be used to solve the linear system :math:`D L = y_{\text{sample}}`.

    Parameters
    ----------
    fit_intercept : bool
        Toggle intercept fitting, default is `True`


    Attributes
    ----------
    weights : NDArray
        Weights of the model after training.
    n_features : int
        Number of features seen in the training stage.
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
        """Fit the model to the training set.

        Parameters
        ----------
        X : ArrayLike, with shape `(n_datapoints, n_features)`
            Training set of datapoints.
        y : ArrayLike, with shape `(n_datapoints, n_target)`
            True target value of the datapoints `X`. Each value associated to a
            datapoint can be a scalar or an array with dimension `n_target`.
        W : ArrayLike, shape can be `[(n_datapoints,), (n_features,), (n_datapoints, n_features)]`
            Weights associated with `X`. Used to weight different datapoints and/or features.

        Returns
        -------
        self : LinearRegression
            Fitted model.
        """
        design_matrix, y = self._pre_process(X, y, W)
        self.weights = np.linalg.pinv(design_matrix) @ y
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
        assert (
            X.shape[1] == self.n_features
        ), f"Expected datapoints to have {self.n_features} but got {X.shape[1]}"
        X = np.column_stack((np.ones(X.shape[0]), X))

        return X @ self.weights

    # TODO: Implement the update method using the Sherman-Morrison formula.
    def update(self, X: ArrayLike, y: ArrayLike, W: ArrayLike):
        """Update the model with a new training set.

        Parameters
        ----------
        X : ArrayLike, shape = `(n_datapoints, n_features)`
            New training set of datapoints.
        y : ArrayLike, shape = `(n_datapoints,)`
            True target value of the datapoints `X`.
        W : ArrayLike, shape can be `[(n_datapoints,), (n_features,), (n_datapoints, n_features)]`
            Weights associated with `X`. Used to weight different datapoints and/or features.

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
        weight_features: bool = False,
    ) -> tuple[NDArray, NDArray]:
        r"""Pre-processing of the dataset.

        Parameters
        ----------
        X : ArrayLike, shape = `(n_datapoints, n_features)`
            Datapoints to be preprocessed.
        y : ArrayLike, shape = `(n_datapoints,)`
            True values associated to `X`.
        W : ArrayLike, shape can be `[(n_datapoints,), (n_features,), (n_datapoints, n_features)]`
            Weights associated with `X`. Used to weight different datapoints and/or features.
        weight_features : bool
            If `X` is a square matrix, we cannot differentiate if the user wants `W` to be a
            weighting of the datapoints or the feature space. By default, the program will assume
            in the square case that `W` is a weighting of the datapoints (that is, `weight_features`
            is defaulted to `False`). For that reason, `weight_features` should be used as a toggle
            to ensure that the program uses the weights in the feature space instead of the
            datapoints. This variable is mostly unused since having a square `X` is a really rare
            case.

        Returns
        -------
        (design_matrix, y) : tuple[NDArray, NDArray]
        """
        X, y = np.asarray(X), np.asarray(y)

        assert X.ndim <= 2, f"Expected X to have at most dimension 2, but got {X.ndim}"

        n_datapoints, self.n_features = X.shape
        assert n_datapoints == len(
            y
        ), f"The lengths of X ({n_datapoints}) and y ({len(y)}) don't match."

        if W is not None:
            W = np.squeeze(np.asarray(W))
            match W.shape, weight_features:
                case (n_datapoints,), False:
                    X = np.einsum("ij,i->ij", X, W)
                case (self.n_features,), _:
                    X = np.einsum("ij,j->ij", X, W)
                case (n_datapoints, self.n_features), _:
                    X = np.einsum("ij,ij->ij", X, W)
                case _:
                    raise RuntimeError(
                        "Weights can have the shapes"
                        f"[({n_datapoints},), ({self.n_features},), ({n_datapoints, self.n_features})],"
                        f"but got {W.shape}"
                    )

        # TODO: Should we really waste time stacking?
        #
        # The padding of ones is applied to the matrix `X` in order to smoothly deal
        # with the weights when taking products of the form `X @ weights`. The padding
        # is applied after the calculating the base function on `X` since we don't want
        # to apply the base function to the bias parameter of the model but solely on
        # the features of the given dataset.
        X = np.column_stack((np.ones(n_datapoints), X))
        self.weights = np.zeros(n_datapoints + 1)

        return X, y
