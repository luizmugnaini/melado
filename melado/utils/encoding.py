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
    enc_classes : NDArray, with shape `(n_samples,)
        Array containing the result of encoding `y` via the one-hot method.
    """

    def __init__(self, y: ArrayLike) -> None:
        y = np.asarray(y)
        assert (
            y.ndim == 1
        ), f"The array of classes {y} is expected to have dimension 1, instead got {y.ndim}"

        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        self.enc_classes = np.zeros(shape=(len(y), self.n_classes))
        if y.dtype.type is np.string_:
            for idx, cl in enumerate(y):
                for pos, val in enumerate(self.classes):
                    if cl == val:
                        self.enc_classes[idx, pos] = 1.0
                        break
        else:
            for idx, cl in enumerate(y):
                self.enc_classes[idx, cl] = 1.0
