import joblib

from src.data.preprocess import (
    preprocess_text
)


TFIDF_PATH = (
    "models/tfidf_vectorizer.pkl"
)

MODEL_PATH = (
    "models/best_model.pkl"
)


# Load trained objects
tfidf = joblib.load(
    TFIDF_PATH
)

model = joblib.load(
    MODEL_PATH
)


def predict_disease(symptoms):
    """
    Predict disease from symptom text.
    """

    # 1. Clean input
    cleaned_text = preprocess_text(
        symptoms
    )

    # 2. Convert text into TF-IDF
    vector = tfidf.transform(
        [cleaned_text]
    )

    # 3. Predict
    prediction = model.predict(
        vector
    )[0]

    return prediction