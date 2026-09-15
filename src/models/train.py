from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def train_logistic_regression(
    X_train,
    y_train
):
    """
    Train Logistic Regression model.
    """

    model = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def train_naive_bayes(
    X_train,
    y_train
):
    """
    Train Multinomial Naive Bayes model.
    """

    model = MultinomialNB()

    model.fit(
        X_train,
        y_train
    )

    return model


def train_linear_svc(
    X_train,
    y_train
):
    """
    Train Linear SVM model.
    """

    model = LinearSVC(
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def train_all_models(
    X_train,
    y_train
):
    """
    Train all classification models.
    """

    lr_model = train_logistic_regression(
        X_train,
        y_train
    )

    nb_model = train_naive_bayes(
        X_train,
        y_train
    )

    svm_model = train_linear_svc(
        X_train,
        y_train
    )

    models = {
        "Logistic Regression": lr_model,
        "Naive Bayes": nb_model,
        "Linear SVC": svm_model
    }

    return models