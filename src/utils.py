"""
utils.py
--------
Evaluation metrics, reporting, and visualisation helpers for the
Heart Disease Prediction project.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend – safe for scripts
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


def evaluate_model(model, X_test: np.ndarray, y_test) -> dict:
    """
    Compute a standard suite of binary-classification metrics.

    Parameters
    ----------
    model  : fitted sklearn estimator with predict / predict_proba
    X_test : np.ndarray
    y_test : array-like

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc
    """
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    metrics = {
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall"   : recall_score(y_test, y_pred, zero_division=0),
        "f1"       : f1_score(y_test, y_pred, zero_division=0),
        "roc_auc"  : roc_auc_score(y_test, y_prob) if y_prob is not None else None,
    }
    return metrics


def print_report(metrics: dict, model_name: str = "Model") -> None:
    """Print a formatted evaluation summary to stdout."""
    sep = "-" * 42
    print(f"\n{sep}")
    print(f"  Evaluation Report — {model_name}")
    print(sep)
    for k, v in metrics.items():
        val = f"{v:.4f}" if v is not None else "N/A"
        print(f"  {k.upper():<12}: {val}")
    print(sep + "\n")


def plot_confusion_matrix(model, X_test, y_test,
                          save_path: str = "outputs/confusion_matrix.png") -> None:
    """
    Plot and save a labelled confusion matrix heatmap.

    Parameters
    ----------
    model     : fitted estimator
    X_test    : np.ndarray
    y_test    : array-like
    save_path : str
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    labels = ["No Disease", "Disease"]
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks); ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(tick_marks); ax.set_yticklabels(labels)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved -> {save_path}")


def plot_roc_curve(model, X_test, y_test,
                   save_path: str = "outputs/roc_curve.png") -> None:
    """
    Plot and save the ROC curve with AUC annotation.

    Parameters
    ----------
    model     : fitted estimator with predict_proba
    X_test    : np.ndarray
    y_test    : array-like
    save_path : str
    """
    if not hasattr(model, "predict_proba"):
        print("[WARN] Model does not support predict_proba – skipping ROC plot.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] ROC curve saved -> {save_path}")


def plot_feature_importance(model, feature_names: list,
                            save_path: str = "outputs/feature_importance.png") -> None:
    """
    Plot feature importances for tree-based models.

    Silently skipped for models that do not expose feature_importances_.

    Parameters
    ----------
    model         : fitted estimator
    feature_names : list[str]
    save_path     : str
    """
    if not hasattr(model, "feature_importances_"):
        print("[INFO] Feature importance plot only available for tree-based models.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(importances)), importances[indices], color="steelblue")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_ylabel("Importance Score")
    ax.set_title("Feature Importances (Gini)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Feature importance plot saved -> {save_path}")
