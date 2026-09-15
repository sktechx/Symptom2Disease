import pandas as pd


def load_dataset(file_path):
    """
    Load the Symptom2Disease dataset.

    Parameters:
        file_path (str): Path to CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """

    try:
        df = pd.read_csv(file_path)

        print(f"Dataset loaded successfully.")
        print(f"Shape: {df.shape}")

        return df

    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}")
        raise

    except Exception as e:
        print(f"Error while loading dataset: {e}")
        raise


if __name__ == "__main__":

    path = "data/raw/Symptom2Disease.csv"

    df = load_dataset(path)

    print(df.head())