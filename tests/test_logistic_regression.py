"""Tests for the logistic regression model."""
# Author: Luiz G. Mugnaini A.
from melado.linear import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import pandas as pd


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
# y = np.trim_zeros(np.select([y == "Iris-setosa", y == "Iris-versicolor"], [-1, 1]))
X = df[["sepal length in cm", "sepal width in cm", "petal length in cm"]].to_numpy()
# X = X[range(len(y))]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


def test_accuracy_score():
    mel_lr = LogisticRegression(epochs=300, batch_size=1).fit(X_train.copy(), y_train.copy())
    mel_pred = mel_lr.predict(X_test)
    assert metrics.accuracy_score(y_test, mel_pred) >= 0.93
