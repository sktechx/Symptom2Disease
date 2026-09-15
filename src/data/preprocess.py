import re
import pandas as pd


def preprocess_text(text):
    """
    Clean input text.

    Steps:
    1. Convert to lowercase
    2. Remove special characters and numbers
    3. Remove extra spaces
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def preprocess_dataframe(df):
    """
    Clean the dataset.

    Required columns:
        text
        label

    Returns:
        Cleaned DataFrame
    """

    df = df.copy()

    # Remove unnecessary unnamed columns
    unnamed_columns = [
        col for col in df.columns
        if col.lower().startswith("unnamed")
    ]

    if unnamed_columns:
        df = df.drop(
            columns=unnamed_columns
        )

    # Remove missing values
    df = df.dropna(
        subset=["text", "label"]
    )

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Clean text
    df["clean_text"] = df["text"].apply(
        preprocess_text
    )

    # Remove empty text
    df = df[
        df["clean_text"].str.len() > 0
    ]

    return df


if __name__ == "__main__":

    input_path = "data/raw/Symptom2Disease.csv"
    output_path = "data/processed/cleaned_data.csv"

    df = pd.read_csv(input_path)

    df = preprocess_dataframe(df)

    df.to_csv(
        output_path,
        index=False
    )

    print("Preprocessing completed.")
    print(f"Cleaned dataset shape: {df.shape}")