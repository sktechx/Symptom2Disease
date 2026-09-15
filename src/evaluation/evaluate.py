import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Evaluate model using:
    Accuracy
    Classification Report
    Confusion Matrix
    """

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(
        f"\nAccuracy: "
        f"{accuracy:.4f}"
    )

    print(
        "\nClassification Report:\n"
    )

    print(report)

    return (
        accuracy,
        report,
        cm,
        predictions
    )


def save_confusion_matrix(
    cm,
    labels,
    output_path
):
    """
    Save confusion matrix as PNG.
    """

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    plt.figure(
        figsize=(14, 10)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.xticks(
        rotation=90
    )

    plt.yticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


def save_model_results(
    results,
    output_path
):
    """
    Save model comparison results
    as JSON.
    """

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    with open(
        output_path,
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )