import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)

import streamlit as st
import joblib

from src.data.preprocess import preprocess_text


# ============================================================
# LOAD MODEL
# ============================================================

TFIDF_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "tfidf_vectorizer.pkl"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "best_model.pkl"
)

tfidf = joblib.load(TFIDF_PATH)
model = joblib.load(MODEL_PATH)


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Symptom → Disease Classifier",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# UI
# ============================================================

st.title("🩺 Symptom → Disease Classifier")

st.write(
    "Enter your symptoms below and the trained ML model "
    "will predict the most likely disease class."
)

st.warning(
    "⚠️ This is an educational ML project and not a medical diagnosis."
)


symptoms = st.text_area(
    "Enter your symptoms:",
    placeholder="Example: fever, headache, body pain, cough...",
    height=150
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("🔍 Predict Disease", use_container_width=True):

    if not symptoms.strip():

        st.error("Please enter your symptoms first.")

    else:

        cleaned_text = preprocess_text(symptoms)

        vector = tfidf.transform([cleaned_text])

        prediction = model.predict(vector)[0]

        st.success("Prediction completed!")

        st.subheader("Predicted Disease")

        st.info(f"🩺 {prediction}")