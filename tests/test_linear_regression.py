"""Tests for the ordinary and weighted linear regression models."""
# Author: Luiz G. Mugnaini A.
import numpy as np
import sklearn.datasets as datasets
import sklearn.metrics as metrics
import sklearn.model_selection as model_selection
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from melado.linear import LinearRegression

# ** Fetch data **

X, y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.20, random_state=1
)

# ** Ordinary linear regression **

# Scikit-learn ordinary linear regression.
sk_olr = SklearnLinearRegression().fit(X_train, y_train)
sk_olr_train_prediction = sk_olr.predict(X_train)
sk_olr_test_prediction = sk_olr.predict(X_test)

# Melado ordinary linear regression.
mel_olr = LinearRegression().fit(X_train, y_train)
mel_olr_train_prediction = mel_olr.predict(X_train)
mel_olr_test_prediction = mel_olr.predict(X_test)


def test_olr_coefficients() -> None:
    sk_olr_coef = sk_olr.coef_
    match sk_olr_coef.ndim:
        case 1:
            sk_olr_coef = np.concatenate((np.array([sk_olr.intercept_]), sk_olr_coef))
        case 2:
            if isinstance(sk_olr.intercept_, np.ndarray):
                sk_olr_coef = np.column_stack((sk_olr.intercept_, sk_olr_coef))

    np.testing.assert_almost_equal(mel_olr.coef, sk_olr_coef)


def test_olr_train_predictions() -> None:
    np.testing.assert_almost_equal(mel_olr_train_prediction, sk_olr_train_prediction)


def test_olr_test_predictions() -> None:
    np.testing.assert_almost_equal(mel_olr_test_prediction, sk_olr_test_prediction)


def test_olr_train_r2_score() -> None:
    mel_olr_train_r2 = metrics.r2_score(y_train, mel_olr_train_prediction)
    sk_olr_train_r2 = metrics.r2_score(y_train, sk_olr_train_prediction)
    np.testing.assert_almost_equal(mel_olr_train_r2, sk_olr_train_r2)


def test_olr_test_r2_score() -> None:
    mel_olr_test_r2 = metrics.r2_score(y_test, mel_olr_test_prediction)
    sk_olr_test_r2 = metrics.r2_score(y_test, sk_olr_test_prediction)
    np.testing.assert_almost_equal(mel_olr_test_r2, sk_olr_test_r2)


# ** Weighted linear regression **

# With previous information on the residuals of the ordinary linear regression
# we can compute the weighted linear regression.
residuals = y_train - sk_olr_train_prediction
weights = residuals**2

mel_wlr = LinearRegression().fit(X_train, y_train, weights)
mel_wlr_train_prediction = mel_wlr.predict(X_train)
mel_wlr_test_prediction = mel_wlr.predict(X_test)

sk_wlr = SklearnLinearRegression().fit(X_train, y_train, weights)
sk_wlr_train_prediction = sk_wlr.predict(X_train)
sk_wlr_test_prediction = sk_wlr.predict(X_test)


def test_wlr_coefficients() -> None:
    sk_wlr_coef = sk_wlr.coef_
    match sk_wlr_coef.ndim:
        case 1:
            sk_wlr_coef = np.concatenate((np.array([sk_wlr.intercept_]), sk_wlr_coef))
        case 2:
            if isinstance(sk_wlr.intercept_, np.ndarray):
                sk_wlr_coef = np.column_stack((sk_wlr.intercept_, sk_wlr_coef))

    np.testing.assert_almost_equal(mel_wlr.coef, sk_wlr_coef)


def test_wlr_train_predictions() -> None:
    np.testing.assert_almost_equal(mel_wlr_train_prediction, sk_wlr_train_prediction)


def test_wlr_test_predictions() -> None:
    np.testing.assert_almost_equal(mel_wlr_test_prediction, sk_wlr_test_prediction)


# def test_wlr_train_r2_score():
#     mel_wlr_train_r2 = metrics.r2_score(y_train, mel_wlr_train_prediction)
#     sk_wlr_train_r2 = metrics.r2_score(y_train, mel_olr_train_prediction)
#     np.testing.assert_almost_equal(mel_wlr_train_r2, sk_wlr_train_r2)
#
#
# def test_wlr_test_r2_score():
#     mel_wlr_test_r2 = metrics.r2_score(y_test, mel_wlr_test_prediction)
#     sk_wlr_test_r2 = metrics.r2_score(y_test, mel_olr_test_prediction)
#     np.testing.assert_almost_equal(mel_wlr_test_r2, sk_wlr_test_r2)
