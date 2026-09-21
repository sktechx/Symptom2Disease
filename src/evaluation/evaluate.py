import os
import json

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

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
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return (
        accuracy,
        report,
        cm,
        predictions
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    cm,
    labels,
    path
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    fig, ax = plt.subplots(
        figsize=(14, 12)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    display.plot(
        ax=ax,
        xticks_rotation=90,
        colorbar=False
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()


# ============================================================
# SAVE RESULTS
# ============================================================

def save_model_results(
    results,
    path
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(
        path,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )