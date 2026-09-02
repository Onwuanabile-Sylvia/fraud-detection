import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """
    Load the credit card fraud dataset from a CSV file.
    """
    df = pd.read_csv(file_path)

    return df


def split_features_and_target(df):
    """
    Separate the predictor variables from the target variable.

    Class is the target.
    Amount_Bin is an EDA-only column and is excluded if present.
    """
    X = df.drop(
        columns=['Class', 'Amount_Bin'],
        errors='ignore'
    )

    y = df['Class']

    return X, y


def split_data(X, y):
    """
    Split the dataset into stratified training and testing sets.

    Uses an 80/20 train-test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Standardize Time and Amount.

    The scaler is fitted only on the training data
    and then used to transform both training and testing data.
    """
    scaler = StandardScaler()

    scaler.fit(
        X_train[['Time', 'Amount']]
    )

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[['Time', 'Amount']] = scaler.transform(
        X_train[['Time', 'Amount']]
    )

    X_test[['Time', 'Amount']] = scaler.transform(
        X_test[['Time', 'Amount']]
    )

    return X_train, X_test, scaler