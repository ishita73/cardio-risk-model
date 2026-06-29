"""
eda.ipynb  (source — rendered as a .py script for clarity)
-----------------------------------------------------------
Exploratory Data Analysis for the UCI Heart Disease dataset.

Sections
--------
1. Data Loading & Overview
2. Univariate Distributions
3. Feature–Target Correlations
4. Pairplot (subset of features)
5. Missing Value Summary

Run from the project root after placing the CSV in data/.
"""

# ── Imports ─────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing import load_data, clean_data

# ── 1. Data Overview ─────────────────────────────────────────────────────────
df_raw = load_data("../data/heart_disease_uci.csv")
print("Raw shape:", df_raw.shape)
print(df_raw.head())
print("\nData types:")
print(df_raw.dtypes)

df = clean_data(df_raw.copy())
print("\nCleaned shape:", df.shape)
print("\nDescriptive statistics:")
print(df.describe().round(2))

# ── 2. Class Balance ─────────────────────────────────────────────────────────
counts = df["target"].value_counts()
fig, ax = plt.subplots(figsize=(4, 4))
ax.bar(["No Disease", "Disease"], counts.values, color=["steelblue", "tomato"])
ax.set_ylabel("Patient Count")
ax.set_title("Class Distribution")
plt.tight_layout()
plt.savefig("../outputs/eda_class_distribution.png", dpi=150)
plt.close()
print("\nSaved: outputs/eda_class_distribution.png")

# ── 3. Correlation Heatmap ───────────────────────────────────────────────────
import matplotlib.patches as mpatches

corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(corr.columns, fontsize=8)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("../outputs/eda_correlation_heatmap.png", dpi=150)
plt.close()
print("Saved: outputs/eda_correlation_heatmap.png")

# ── 4. Feature Distributions by Class ───────────────────────────────────────
continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

fig, axes = plt.subplots(1, len(continuous_features), figsize=(16, 4))
for ax, feat in zip(axes, continuous_features):
    for cls, colour, label in [(0, "steelblue", "No Disease"), (1, "tomato", "Disease")]:
        subset = df[df["target"] == cls][feat]
        ax.hist(subset, bins=20, alpha=0.6, color=colour, label=label, density=True)
    ax.set_title(feat)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)

plt.suptitle("Feature Distributions by Class", y=1.02)
plt.tight_layout()
plt.savefig("../outputs/eda_feature_distributions.png", dpi=150)
plt.close()
print("Saved: outputs/eda_feature_distributions.png")
