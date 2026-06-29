"""
preprocessing.py
----------------
Data loading, cleaning, and preprocessing pipeline for the
Heart Disease Prediction project.

Dataset: UCI Heart Disease (Cleveland subset)
Source : https://archive.ics.uci.edu/dataset/45/heart+disease
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw heart disease dataset from a CSV file.

    Handles both named-column CSVs (e.g. kaggle version) and
    raw UCI-format files with no header row.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    pd.DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        if df.shape[1] == 14 and list(df.columns) != COLUMN_NAMES:
            df.columns = COLUMN_NAMES
    except Exception as e:
        raise IOError(f"Failed to load data from '{filepath}': {e}")

    print(f"[INFO] Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and binarise the target variable.

    The UCI dataset uses '?' for missing entries. This function
    replaces them with NaN, drops incomplete rows, and converts
    the multi-class target (0-4) to binary:
        0 -> no disease
        1 -> disease present (original values 1-4)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df.replace("?", np.nan, inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_before = len(df)
    df.dropna(inplace=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[INFO] Dropped {n_dropped} rows containing missing values")

    df["target"] = (df["target"] > 0).astype(int)
    no_disease = (df["target"] == 0).sum()
    disease    = (df["target"] == 1).sum()
    print(f"[INFO] Class distribution -> No Disease: {no_disease} | Disease: {disease}")

    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame):
    """Return feature matrix X and target vector y."""
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def scale_features(X_train, X_test):
    """
    Fit StandardScaler on training data and apply to both splits.
    Prevents data leakage by never fitting on test data.

    Returns
    -------
    X_train_scaled, X_test_scaled : np.ndarray
    scaler                        : fitted StandardScaler instance
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def preprocess(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """
    Complete preprocessing pipeline: load -> clean -> split -> scale.

    Parameters
    ----------
    filepath     : str   path to the raw CSV
    test_size    : float fraction of data reserved for testing
    random_state : int

    Returns
    -------
    X_train, X_test : np.ndarray
    y_train, y_test : pd.Series
    scaler          : fitted StandardScaler
    feature_names   : list[str]
    """
    df = load_data(filepath)
    df = clean_data(df)
    X, y = split_features_target(df)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y            # preserve class ratio in both splits
    )

    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
    print(f"[INFO] Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    return X_train_sc, X_test_sc, y_train, y_test, scaler, feature_names
