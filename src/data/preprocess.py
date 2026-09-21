import re
import pandas as pd


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    if pd.isna(text):

        return ""

    text = str(text).lower()

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# DATAFRAME PREPROCESSING
# ============================================================

def preprocess_dataframe(df):

    df = df.copy()

    # Make sure required columns exist
    if "text" not in df.columns:

        raise ValueError(
            "Dataset must contain a 'text' column."
        )

    if "label" not in df.columns:

        raise ValueError(
            "Dataset must contain a 'label' column."
        )

    # Clean text
    df["clean_text"] = df["text"].apply(
        preprocess_text
    )

    # Remove empty rows
    df = df[
        df["clean_text"].str.len() > 0
    ]

    return df