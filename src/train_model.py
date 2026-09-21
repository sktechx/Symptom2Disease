import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from src.data.load_data import load_dataset
from src.data.preprocess import preprocess_dataframe

from src.features.entities import (
    create_entity_recognizer,
    extract_entities,
    extract_symptoms
)

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


# ============================================================
# PATHS
# ============================================================

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

ENTITY_PATH = (
    "data/processed/entity_data.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "SYMPTOM → DISEASE CLASSIFIER"
    )

    print(
        "NER + TF-IDF + ML"
    )

    print("=" * 60)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    df = load_dataset(
        DATA_PATH
    )

    print(
        "\nOriginal Dataset:"
    )

    print(
        df.head()
    )

    # ========================================================
    # 2. PREPROCESS DATA
    # ========================================================

    df = preprocess_dataframe(
        df
    )

    print(
        f"\nAfter preprocessing: "
        f"{df.shape}"
    )

    # ========================================================
    # 3. CREATE ENTITY RECOGNIZER
    # ========================================================

    print("\nCreating Entity Recognizer...")

    nlp = create_entity_recognizer()

    # ========================================================
    # 4. ENTITY EXTRACTION
    # ========================================================

    print(
        "\nExtracting symptoms..."
    )

    df["entities"] = df[
        "clean_text"
    ].apply(
        lambda text: extract_entities(
            text,
            nlp
        )
    )

    df["extracted_symptoms"] = df[
        "clean_text"
    ].apply(
        lambda text: extract_symptoms(
            text,
            nlp
        )
    )

    # ========================================================
    # 5. CONVERT ENTITIES TO TEXT
    # ========================================================

    df["entity_text"] = df[
        "extracted_symptoms"
    ].apply(
        lambda symptoms:
        " ".join(symptoms)
    )

    # ========================================================
    # 6. SHOW ENTITY EXAMPLES
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "ENTITY EXTRACTION EXAMPLES"
    )

    print(
        "=" * 60
    )

    for i in range(
        min(10, len(df))
    ):

        print(
            f"\nOriginal: "
            f"{df.iloc[i]['text']}"
        )

        print(
            f"Entities: "
            f"{df.iloc[i]['entities']}"
        )

        print(
            f"Symptoms: "
            f"{df.iloc[i]['extracted_symptoms']}"
        )

    # ========================================================
    # 7. SAVE ENTITY DATA
    # ========================================================

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    df[
        [
            "text",
            "label",
            "clean_text",
            "entity_text"
        ]
    ].to_csv(
        ENTITY_PATH,
        index=False
    )

    print(
        f"\nEntity data saved to: "
        f"{ENTITY_PATH}"
    )

    # ========================================================
    # 8. FEATURES + TARGET
    # ========================================================

    X = df[
        "entity_text"
    ]

    y = df[
        "label"
    ]

    # Remove samples where no symptoms were detected
    valid_rows = X.str.len() > 0

    X = X[
        valid_rows
    ]

    y = y[
        valid_rows
    ]

    print(
        f"\nSamples after entity extraction: "
        f"{len(X)}"
    )

    # ========================================================
    # 9. TRAIN TEST SPLIT
    # ========================================================

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    print(
        f"\nTraining samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )

    # ========================================================
    # 10. TF-IDF
    # ========================================================

    tfidf = (
        create_tfidf_vectorizer()
    )

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

    # ========================================================
    # 11. TRAIN MODELS
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "TRAINING MODELS"
    )

    print(
        "=" * 60
    )

    models = train_all_models(

        X_train_tfidf,

        y_train
    )

    # ========================================================
    # 12. MODEL COMPARISON
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "=" * 60
    )

    results = compare_models(

        models,

        X_test_tfidf,

        y_test
    )

    # ========================================================
    # 13. BEST MODEL
    # ========================================================

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

    # ========================================================
    # 14. CREATE MODEL DIRECTORY
    # ========================================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # ========================================================
    # 15. SAVE ENTITY RECOGNIZER
    # ========================================================

    entity_path = (
        f"{MODEL_DIR}/entity_recognizer"
    )

    nlp.to_disk(
        entity_path
    )

    print(
        f"\nEntity recognizer saved: "
        f"{entity_path}"
    )

    # ========================================================
    # 16. SAVE TF-IDF
    # ========================================================

    joblib.dump(

        tfidf,

        f"{MODEL_DIR}/tfidf_vectorizer.pkl"
    )

    # ========================================================
    # 17. SAVE INDIVIDUAL MODELS
    # ========================================================

    joblib.dump(

        models[
            "Logistic Regression"
        ],

        f"{MODEL_DIR}/logistic_regression.pkl"
    )

    joblib.dump(

        models[
            "Naive Bayes"
        ],

        f"{MODEL_DIR}/naive_bayes.pkl"
    )

    joblib.dump(

        models[
            "Linear SVC"
        ],

        f"{MODEL_DIR}/linear_svc.pkl"
    )

    # ========================================================
    # 18. SAVE BEST MODEL
    # ========================================================

    joblib.dump(

        best_model,

        f"{MODEL_DIR}/best_model.pkl"
    )

    # ========================================================
    # 19. EVALUATE BEST MODEL
    # ========================================================

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

    # ========================================================
    # 20. CONFUSION MATRIX
    # ========================================================

    labels = sorted(
        y.unique()
    )

    save_confusion_matrix(

        cm,

        labels,

        CM_PATH
    )

    # ========================================================
    # 21. SAVE RESULTS
    # ========================================================

    save_model_results(

        results,

        RESULTS_PATH
    )

    # ========================================================
    # 22. FINAL OUTPUT
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "TRAINING COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 60
    )

    print(
        f"\nBest Model: "
        f"{best_model_name}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        "\nSaved Files:"
    )

    print(
        "✓ Entity Recognizer"
    )

    print(
        "✓ TF-IDF Vectorizer"
    )

    print(
        "✓ Logistic Regression"
    )

    print(
        "✓ Naive Bayes"
    )

    print(
        "✓ Linear SVC"
    )

    print(
        "✓ Best Model"
    )

    print(
        "✓ Confusion Matrix"
    )

    print(
        "✓ Model Results"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()