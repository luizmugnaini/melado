import numpy as np
from melado.utils import OneHotEncoder


def test_one_hot_str():
    classes = ["Link", "Zelda", "Hylia", "Ganondorf"]
    y = np.array(["Link", "Link", "Ganondorf", "Hylia", "Ganondorf", "Zelda", "Hylia", "Link"])
    encoded = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    enc = OneHotEncoder(y)
    assert enc.n_classes == len(classes), "Number of classes don't match."
    assert set(enc.classes) == set(classes), "Classes don't match"
    np.testing.assert_equal(encoded.astype(int), enc.encoded.astype(int))


def test_one_hot_numbers():
    classes = [1, 3, 45, 6]
    y = np.array([1, 1, 3, 45, 6, 3, 45])
    encoded = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    enc = OneHotEncoder(y)
    assert enc.n_classes == len(classes), "Number of classes don't match."
    assert set(enc.classes) == set(classes), "Classes don't match"
    np.testing.assert_equal(encoded.astype(int), enc.encoded.astype(int))
