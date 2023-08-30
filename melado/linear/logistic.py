"""Module implementing the logistic regression for classification tasks."""
# Author: Luiz G. Mugnaini A.
from ..utils import OneHotEncoder
import numpy as np
from typing import Optional, Self
from numpy.typing import ArrayLike, NDArray
import logging
import sys

logger = logging.getLogger("logistic-regression")
handle = logging.StreamHandler(sys.stderr)
fmt = logging.Formatter("[%(levelname)s] %(message)s")
handle.setFormatter(fmt)
logger.addHandler(handle)
logger.setLevel(logging.DEBUG)


class LogisticRegression:
    """Logistic classification ("regression") model.

    Parameters
    ----------
    learning_rate: float (optional, defaults to `0.01`)
        How intense will be the effect of the gradient descent method for a single iteration.
        Should be a float between `0.0` and `1.0`.
    tol: float | None (optional, defaults to `1e-4`)
        Minimum accepted error tolerance.
    epochs : int (optional, defaults to `100`)
        The number of iterations the algorithm should be ran.
    l2 : float (defaults to `0.0`)
        L2 regularization.
    random_state : int | None (optional, defaults to `None`)
        Random state used for weight initialization.
    batch_size : int (defaults to `200`)
        The number of samples to be used per batch.
    """

    __slots__ = (
        "learning_rate",
        "l2",
        "epochs",
        "fit_intercept",
        "random_state",
        "batch_size",
        "shuffle",
        "is_fit",
        "n_classes",
        "classes",
        "n_features",
        "weights",
        "epoch_cost",
    )

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 100,
        l2: float = 0.0,
        fit_intercept: bool = True,
        random_state: Optional[int] = 1,
        batch_size: int = 100,
        shuffle: bool = True,
    ) -> None:
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.is_fit = False
        self.n_classes: Optional[int] = None
        self.classes: Optional[NDArray] = None
        self.n_features: Optional[int] = None
        self.weights: Optional[NDArray] = None
        self.epoch_cost: Optional[list] = None

    @staticmethod
    def _softmax(net_input: NDArray) -> NDArray:
        if net_input.ndim > 1:
            return (np.exp(net_input.T) / np.sum(np.exp(net_input), axis=1)).T
        else:
            return (np.exp(net_input.T) / np.sum(np.exp(net_input))).T

    def _forward(self, X: NDArray) -> NDArray:
        net_input = (
            X.dot(self.weights[1:]) + self.weights[0]  # pyre-ignore[16]
            if self.fit_intercept
            else X.dot(self.weights)  # pyre-ignore[6]
        )
        return LogisticRegression._softmax(net_input)

    def _cost(self, actual: NDArray, expected: NDArray) -> float:
        l2 = self.l2 * np.sum(self.weights**2)  # pyre-ignore[58]
        cross_entropy = -np.sum(np.log(actual) * expected, axis=1) + l2
        return 0.5 * np.mean(cross_entropy)  # pyre-ignore[58]

    def fit(self, X: ArrayLike, y: ArrayLike, hot_start: bool = False) -> Self:
        """Fit training data to the model."""
        if (not self.is_fit) and hot_start:
            logger.warning("Ignoring hot start argument, the model isn't fitted yet.")

        X, y = np.asarray(X), np.asarray(y)
        n_datapoints, self.n_features = X.shape
        assert n_datapoints == len(y), f"Lengths of X ({n_datapoints}) and y ({len(y)}) don't match."
        assert y.ndim == 1, f"Expected y to have dimension 1, instead got {y.ndim}."

        enc = OneHotEncoder(y)
        y_enc = enc.encoded
        self.classes = enc.classes
        self.n_classes = enc.n_classes

        if isinstance(self.random_state, int):
            np.random.seed(self.random_state)

        self.weights = np.random.normal(
            loc=0.0,
            scale=0.1,
            size=(
                self.n_features + 1 if self.fit_intercept else self.n_features,  # pyre-ignore[58]
                self.n_classes,
            ),
        )

        self.epoch_cost = []
        epoch_idx = np.arange(n_datapoints)
        for _ in range(self.epochs):
            if self.shuffle:
                np.random.shuffle(epoch_idx)

            for fst_idx in range(0, n_datapoints - self.batch_size + 1, self.batch_size):
                batch_idx = epoch_idx[fst_idx : fst_idx + self.batch_size]
                X_batch = X[batch_idx]
                y_true = y_enc[batch_idx]

                y_prob = self._forward(X_batch)

                # Gradient descent
                out_diff = y_true - y_prob
                grad_weights = X_batch.T.dot(out_diff)
                reg_coeff = self.l2 * self.weights[1:]  # pyre-ignore[16]
                if self.fit_intercept:
                    # pyre-ignore[16]
                    self.weights[0] += self.learning_rate * np.sum(out_diff, axis=0)
                    self.weights[1:] += self.learning_rate * (grad_weights - reg_coeff)
                else:
                    # pyre-ignore[16]
                    self.weights += self.learning_rate * (grad_weights - reg_coeff)

            y_prob_epoch = self._forward(X)
            # pyre-ignore[16]
            self.epoch_cost.append(self._cost(actual=y_prob_epoch, expected=y_enc))

        return self

    def predict_prob(self, X: NDArray) -> NDArray:
        if X.ndim > 1:
            assert (
                X.shape[1] == self.n_features
            ), f"Number of features don't match: expected {self.n_features}, got {X.shape[1]}"
        return self._forward(X)

    def predict(self, X: NDArray) -> NDArray:
        prob = self.predict_prob(X)
        return np.array(
            [self.classes[max_idx] for max_idx in prob.argmax(axis=1)]  # pyre-ignore[16]
        )
