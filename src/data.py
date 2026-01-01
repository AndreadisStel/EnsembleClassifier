import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_split_data(file_path, test_size=0.2, random_state=42):
    """
    Load dataset from CSV and split into training and validation sets.
    """

    # No header
    df = pd.read_csv(file_path, header=None)

    # last column -> label
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_val, y_train, y_val


def load_full_training_data(file_path):
    """
    Load the full labeled dataset without splitting.
    Used for final training before test prediction.
    """

    # No header
    df = pd.read_csv(file_path, header=None)
    # last column -> label
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


def load_test_data(file_path) -> pd.DataFrame:
    """
    Load unlabeled test dataset as dataframe.
    """
    # No header
    df = pd.read_csv(file_path, header=None)
    return df

