import sys
import os

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
import joblib

from src.data.preprocess import preprocess_text

from src.features.entities import (
    create_entity_recognizer,
    extract_symptoms
)


# ============================================================
# PATHS
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

ENTITY_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "entity_recognizer"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    # Load TF-IDF
    tfidf = joblib.load(
        TFIDF_PATH
    )

    # Load best ML model
    model = joblib.load(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Load Entity Recognizer
    # --------------------------------------------------------

    if os.path.exists(
        ENTITY_MODEL_PATH
    ):

        import spacy

        nlp = spacy.load(
            ENTITY_MODEL_PATH
        )

    else:

        nlp = create_entity_recognizer()

    return (
        tfidf,
        model,
        nlp
    )


tfidf, model, nlp = load_models()


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(

    page_title="Symptom → Disease Classifier",

    page_icon="🩺",

    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main title */

    .main-title {

        font-size: 38px;

        font-weight: 700;

        text-align: center;

        margin-bottom: 5px;
    }


    /* Subtitle */

    .subtitle {

        text-align: center;

        color: #666;

        font-size: 17px;

        margin-bottom: 25px;
    }


    /* Prediction box */

    .prediction-box {

        padding: 20px;

        border-radius: 12px;

        background-color: #eef8ee;

        border-left: 6px solid #28a745;

        text-align: center;

        margin-top: 15px;

    }


    /* Prediction disease */

    .prediction-disease {

        font-size: 28px;

        font-weight: 700;

        color: #000000;

    }


    /* Symptom badge */

    .symptom-badge {

        display: inline-block;

        padding: 8px 14px;

        margin: 5px;

        border-radius: 20px;

        background-color: #f0f7ff;

        border: 1px solid #2196F3;

        color: #000000;

        font-size: 15px;

    }


    /* Footer */

    .footer {

        text-align: center;

        color: #777;

        font-size: 13px;

        margin-top: 25px;

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🩺 Symptom → Disease Classifier
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Enter your symptoms to predict the disease class
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# WARNING
# ============================================================

st.warning(
    "⚠️ This is an educational ML project and "
    "not a medical diagnosis."
)


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader(
    "📝 Enter Your Symptoms"
)

symptoms = st.text_area(

    "Describe your symptoms below:",

    placeholder=(
        "Example: I have fever, cough, "
        "headache and fatigue..."
    ),

    height=160,

    label_visibility="collapsed"
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(

    "🔍 Analyze Symptoms",

    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CHECK EMPTY INPUT
    # --------------------------------------------------------

    if not symptoms.strip():

        st.error(
            "Please enter your symptoms first."
        )

    else:

        # ----------------------------------------------------
        # PREPROCESS TEXT
        # ----------------------------------------------------

        cleaned_text = preprocess_text(
            symptoms
        )


        # ----------------------------------------------------
        # ENTITY EXTRACTION
        # ----------------------------------------------------

        extracted_symptoms = extract_symptoms(
            cleaned_text,
            nlp
        )


        # ----------------------------------------------------
        # CREATE MODEL INPUT
        # ----------------------------------------------------

        entity_text = " ".join(
            extracted_symptoms
        )


        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        # If no predefined entity was found,
        # use the complete cleaned text.

        if not entity_text.strip():

            entity_text = cleaned_text


        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        vector = tfidf.transform(
            [entity_text]
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            vector
        )[0]


        # ====================================================
        # SUCCESS MESSAGE
        # ====================================================

        st.success(
            "Analysis completed successfully!"
        )


        # ====================================================
        # EXTRACTED SYMPTOMS
        # ====================================================

        if extracted_symptoms:

            st.subheader(
                "🩺 Extracted Symptoms"
            )

            symptoms_html = ""

            for symptom in extracted_symptoms:

                symptoms_html += (
                    f'<span class="symptom-badge">'
                    f'{symptom}'
                    f'</span>'
                )

            st.markdown(
                symptoms_html,
                unsafe_allow_html=True
            )

        else:

            st.info(
                "No predefined symptoms were detected. "
                "The entered text was used for prediction."
            )


        # ====================================================
        # DISEASE PREDICTION
        # ====================================================

        st.divider()

        st.subheader("🎯 Disease Prediction")

        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="prediction-disease">
                    🩺 {prediction}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        Symptom → Disease Classifier
        <br>
        NLP • Entity Recognition • TF-IDF • Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)

