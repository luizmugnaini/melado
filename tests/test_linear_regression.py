# Author: Luiz G. Mugnaini A.
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
from melado.linear import LinearRegression
import numpy as np

# TODO:
# - To ensure that the model is performing well with respect to the scikit-learn lib, we should do more extensive testing.
# - Implement tests for the weighted linear regression.
# - Implement unit testing.

data = pd.read_csv(
    "/home/mug/Projects/melado/data/linear_regression/ww2_weather/summary_of_weather.csv",
    low_memory=False,
)
X = data[["MinTemp"]].to_numpy().reshape(-1, 1)
y = data["MaxTemp"].to_numpy().reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


def test_linear_regression_r2_score():
    sklearn_lrm = SklearnLinearRegression()
    sklearn_lrm.fit(X_train, y_train)
    sklearn_prediction = sklearn_lrm.predict(X_test)

    melado_lrm = LinearRegression()
    melado_lrm.fit(X_train, y_train)
    melado_prediction = melado_lrm.predict(X_test)
    np.testing.assert_almost_equal(
        r2_score(y_test, sklearn_prediction), r2_score(y_test, melado_prediction)
    )
