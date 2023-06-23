import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron as SkPerceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from melado.linear.perceptron import Perceptron


df = pd.read_csv(
    filepath_or_buffer="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
)
df.columns = [
    "sepal length in cm",
    "sepal width in cm",
    "petal length in cm",
    "petal width in cm",
    "class label",
]
y = df["class label"].to_numpy()
y = np.trim_zeros(np.select([y == "Iris-setosa", y == "Iris-versicolor"], [-1, 1]))
X = df[["sepal length in cm", "petal length in cm"]].to_numpy()
X = X[range(len(y))]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


def test_perceptron_f1_score():
    sklearn_perceptron = SkPerceptron()
    sklearn_perceptron.fit(X_train, y_train)
    sklearn_prediction = sklearn_perceptron.predict(X_test)
    sklearn_f1 = f1_score(y_test, sklearn_prediction)

    melado_perceptron = Perceptron()
    melado_perceptron.fit(X_train, y_train)
    melado_prediction = melado_perceptron.predict(X_test)
    melado_f1 = f1_score(y_test, melado_prediction)
    np.testing.assert_almost_equal(melado_f1, sklearn_f1)
