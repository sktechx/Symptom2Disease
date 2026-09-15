from sklearn.datasets import (
    make_classification
)

from src.models.train import (
    train_logistic_regression,
    train_naive_bayes,
    train_linear_svc
)


def test_logistic_regression():

    X, y = make_classification(
        n_samples=100,
        n_features=10,
        random_state=42
    )

    model = train_logistic_regression(
        X,
        y
    )

    predictions = model.predict(X)

    assert len(predictions) == 100


def test_naive_bayes():

    X, y = make_classification(
        n_samples=100,
        n_features=10,
        random_state=42
    )

    model = train_naive_bayes(
        X,
        y
    )

    predictions = model.predict(X)

    assert len(predictions) == 100


def test_linear_svc():

    X, y = make_classification(
        n_samples=100,
        n_features=10,
        random_state=42
    )

    model = train_linear_svc(
        X,
        y
    )

    predictions = model.predict(X)

    assert len(predictions) == 100