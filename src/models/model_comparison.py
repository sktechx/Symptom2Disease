from sklearn.metrics import accuracy_score


def compare_models(
    models,
    X_test,
    y_test
):
    """
    Compare multiple models using accuracy.
    """

    results = {}

    for name, model in models.items():

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        results[name] = accuracy

        print(
            f"{name}: "
            f"{accuracy:.4f}"
        )

    return results


def get_best_model(
    models,
    results
):
    """
    Select model with highest accuracy.
    """

    best_model_name = max(
        results,
        key=results.get
    )

    best_model = models[
        best_model_name
    ]

    best_accuracy = results[
        best_model_name
    ]

    return (
        best_model_name,
        best_model,
        best_accuracy
    )