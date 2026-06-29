"""
main.py
-------
CLI entry point for the Heart Disease Prediction pipeline.

Usage
-----
    python main.py --data data/heart_disease_uci.csv --model random_forest
    python main.py --data data/heart_disease_uci.csv --model logistic_regression
    python main.py --data data/heart_disease_uci.csv --model svm
"""

import argparse
import os
import sys

from src.preprocessing import preprocess
from src.model import get_model, train_model, save_model
from src.utils import (
    evaluate_model,
    print_report,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
)


BANNER = """
╔══════════════════════════════════════════════════════╗
║       Heart Disease Prediction — ML Pipeline         ║
║       Dataset : UCI Heart Disease (Cleveland)        ║
╚══════════════════════════════════════════════════════╝
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a heart disease prediction model."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/heart_disease_uci.csv",
        help="Path to the input CSV file (default: data/heart_disease_uci.csv)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["logistic_regression", "random_forest", "svm"],
        help="Classifier to train (default: random_forest)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving model artefacts to disk"
    )
    return parser.parse_args()


def main():
    print(BANNER)
    args = parse_args()

    # ── 1. Data validation ─────────────────────────────────────────────────
    if not os.path.isfile(args.data):
        print(f"[ERROR] Dataset not found: {args.data}")
        print("        Place the CSV in the data/ folder or pass --data <path>")
        sys.exit(1)

    # ── 2. Preprocessing ───────────────────────────────────────────────────
    print("\n[STEP 1/4] Preprocessing data ...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(
        filepath=args.data,
        test_size=args.test_size,
        random_state=args.seed,
    )

    # ── 3. Model training ──────────────────────────────────────────────────
    print(f"\n[STEP 2/4] Training model: {args.model} ...")
    clf = get_model(args.model)
    clf = train_model(clf, X_train, y_train)

    # ── 4. Evaluation ──────────────────────────────────────────────────────
    print("\n[STEP 3/4] Evaluating on held-out test set ...")
    metrics = evaluate_model(clf, X_test, y_test)
    print_report(metrics, model_name=args.model.replace("_", " ").title())

    # ── 5. Visualisations & artefacts ──────────────────────────────────────
    print("[STEP 4/4] Generating output plots ...")
    plot_confusion_matrix(clf, X_test, y_test)
    plot_roc_curve(clf, X_test, y_test)
    plot_feature_importance(clf, feature_names)

    if not args.no_save:
        save_model(clf, scaler)

    print("\n[DONE] Pipeline completed successfully.")
    print("       Results saved to outputs/  |  Model saved to models/")


if __name__ == "__main__":
    main()
