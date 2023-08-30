from numpy.typing import ArrayLike
import numpy as np


class OneHotEncoder:
    """One-hot encoder.

    Performs multiclass label encoding via the one-hot method.

    Parameters
    ----------
    y : ArrayLike, with shape `(n_samples,)`
        Array containing the classes to be encoded.

    Attributes
    ----------
    n_classes : int
        Total number of unique classes found in the input array `y`.
    classes : NDArray, with shape `(n_classes,)`
        Array containing the classes present in the input array `y`.
    encoded : NDArray, with shape `(n_samples,)
        Array containing the result of encoding `y` via the one-hot method.
    map : dict
        Dictionary containing the mapping of each class to its corresponding encoding.
    """

    __slots__ = ("classes", "n_classes", "encoded", "map")

    def __init__(self, y: ArrayLike, dtype=np.float64) -> None:
        y = np.asarray(y)
        assert (
            y.ndim == 1
        ), f"The array of classes {y} is expected to have dimension 1, instead got {y.ndim}"

        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        mat = np.eye(self.n_classes, dtype=dtype)
        self.encoded = np.zeros(shape=(len(y), self.n_classes))
        self.map = dict(((self.classes[i], mat[i]) for i in range(self.n_classes)))
        for idx, cl in enumerate(y):
            self.encoded[idx] = self.map[cl]
