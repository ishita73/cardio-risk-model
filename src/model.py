"""
model.py
--------
Model definitions, training, and serialisation for the
Heart Disease Prediction project.

All classifiers are trained with the same interface so they can
be swapped or compared without changing downstream code.
"""

import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# ── Available model registry ────────────────────────────────────────────────
MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="lbfgs"
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "svm": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ),
}


def get_model(model_name: str):
    """
    Retrieve a classifier instance by name from the registry.

    Parameters
    ----------
    model_name : str
        One of 'logistic_regression', 'random_forest', 'svm'.

    Returns
    -------
    sklearn estimator
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]


def train_model(model, X_train: np.ndarray, y_train):
    """
    Fit a classifier on the training data.

    Parameters
    ----------
    model   : sklearn estimator
    X_train : np.ndarray  scaled feature matrix
    y_train : array-like  binary labels

    Returns
    -------
    Fitted estimator
    """
    print(f"[INFO] Training {model.__class__.__name__} ...")
    model.fit(X_train, y_train)
    print("[INFO] Training complete.")
    return model


def save_model(model, scaler, path: str = "models/"):
    """
    Persist trained model and scaler as pickle files.

    Parameters
    ----------
    model  : fitted sklearn estimator
    scaler : fitted StandardScaler
    path   : str  directory where artefacts are saved
    """
    os.makedirs(path, exist_ok=True)
    model_path  = os.path.join(path, "model.pkl")
    scaler_path = os.path.join(path, "scaler.pkl")

    with open(model_path,  "wb") as f:
        pickle.dump(model,  f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[INFO] Model  saved -> {model_path}")
    print(f"[INFO] Scaler saved -> {scaler_path}")


def load_model(path: str = "models/"):
    """
    Load a previously saved model and scaler from disk.

    Parameters
    ----------
    path : str  directory containing model.pkl and scaler.pkl

    Returns
    -------
    model, scaler
    """
    model_path  = os.path.join(path, "model.pkl")
    scaler_path = os.path.join(path, "scaler.pkl")

    with open(model_path,  "rb") as f:
        model  = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    print(f"[INFO] Loaded model  from {model_path}")
    print(f"[INFO] Loaded scaler from {scaler_path}")
    return model, scaler
