import pandas as pd


def load_dataset(path):

    print(
        f"\nLoading dataset from: {path}"
    )

    df = pd.read_csv(path)

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        "\nColumns:"
    )

    print(
        df.columns.tolist()
    )

    return df