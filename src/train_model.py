import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from src.data.load_data import load_dataset
from src.data.preprocess import preprocess_dataframe
from src.features.tfidf_features import (
    create_tfidf_vectorizer,
    fit_tfidf
)
from src.models.train import train_all_models
from src.models.model_comparison import (
    compare_models,
    get_best_model
)
from src.evaluation.evaluate import (
    evaluate_model,
    save_confusion_matrix,
    save_model_results
)


DATA_PATH = (
    "data/raw/Symptom2Disease.csv"
)

MODEL_DIR = "models"

RESULTS_PATH = (
    "reports/metrics/model_results.json"
)

CM_PATH = (
    "reports/figures/confusion_matrix.png"
)


def main():

    print("=" * 60)
    print("SYMPTOM → DISEASE CLASSIFIER")
    print("MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------

    df = load_dataset(
        DATA_PATH
    )

    # --------------------------------------------------
    # 2. PREPROCESS DATA
    # --------------------------------------------------

    df = preprocess_dataframe(
        df
    )

    print(
        f"\nAfter preprocessing: "
        f"{df.shape}"
    )

    # Save cleaned data
    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    df.to_csv(
        "data/processed/cleaned_data.csv",
        index=False
    )

    # --------------------------------------------------
    # 3. FEATURES AND TARGET
    # --------------------------------------------------

    X = df["clean_text"]
    y = df["label"]

    # --------------------------------------------------
    # 4. TRAIN TEST SPLIT
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    print(
        f"\nTraining samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )

    # --------------------------------------------------
    # 5. TF-IDF
    # --------------------------------------------------

    tfidf = create_tfidf_vectorizer()

    (
        X_train_tfidf,
        X_test_tfidf
    ) = fit_tfidf(
        tfidf,
        X_train,
        X_test
    )

    print(
        f"\nTF-IDF train shape: "
        f"{X_train_tfidf.shape}"
    )

    print(
        f"TF-IDF test shape: "
        f"{X_test_tfidf.shape}"
    )

    # --------------------------------------------------
    # 6. TRAIN MODELS
    # --------------------------------------------------

    models = train_all_models(
        X_train_tfidf,
        y_train
    )

    # --------------------------------------------------
    # 7. COMPARE MODELS
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    results = compare_models(
        models,
        X_test_tfidf,
        y_test
    )

    # --------------------------------------------------
    # 8. GET BEST MODEL
    # --------------------------------------------------

    (
        best_model_name,
        best_model,
        best_accuracy
    ) = get_best_model(
        models,
        results
    )

    print(
        f"\nBest Model: "
        f"{best_model_name}"
    )

    print(
        f"Best Accuracy: "
        f"{best_accuracy:.4f}"
    )

    # --------------------------------------------------
    # 9. CREATE MODEL DIRECTORY
    # --------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # --------------------------------------------------
    # 10. SAVE TF-IDF
    # --------------------------------------------------

    joblib.dump(
        tfidf,
        f"{MODEL_DIR}/tfidf_vectorizer.pkl"
    )

    # --------------------------------------------------
    # 11. SAVE INDIVIDUAL MODELS
    # --------------------------------------------------

    joblib.dump(
        models["Logistic Regression"],
        f"{MODEL_DIR}/logistic_regression.pkl"
    )

    joblib.dump(
        models["Naive Bayes"],
        f"{MODEL_DIR}/naive_bayes.pkl"
    )

    joblib.dump(
        models["Linear SVC"],
        f"{MODEL_DIR}/linear_svc.pkl"
    )

    # --------------------------------------------------
    # 12. SAVE BEST MODEL
    # --------------------------------------------------

    joblib.dump(
        best_model,
        f"{MODEL_DIR}/best_model.pkl"
    )

    # --------------------------------------------------
    # 13. EVALUATE BEST MODEL
    # --------------------------------------------------

    (
        accuracy,
        report,
        cm,
        predictions
    ) = evaluate_model(
        best_model,
        X_test_tfidf,
        y_test
    )

    # --------------------------------------------------
    # 14. SAVE CONFUSION MATRIX
    # --------------------------------------------------

    labels = sorted(
        y.unique()
    )

    save_confusion_matrix(
        cm,
        labels,
        CM_PATH
    )

    # --------------------------------------------------
    # 15. SAVE RESULTS
    # --------------------------------------------------

    save_model_results(
        results,
        RESULTS_PATH
    )

    print("\n")
    print("=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"\nBest Model: "
        f"{best_model_name}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )


if __name__ == "__main__":
    main()