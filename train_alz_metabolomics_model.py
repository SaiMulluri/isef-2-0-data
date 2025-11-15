"""Train Alzheimer vs control classifiers on the processed metabolomics dataset.

This script expects :mod:`process_metabolomics.py` to have produced
``data/processed/X_metabolomics.csv`` and ``data/processed/y_metabolomics.csv``.
It loads the feature matrix, splits the data into stratified train/test folds,
fits a baseline RandomForest classifier (plus an optional Logistic Regression
baseline for comparison), evaluates common metrics, and stores diagnostics for
ISEF poster figures.

Evaluation outputs
------------------
* Accuracy, ROC AUC, confusion matrix, and a full classification report printed
  to the console.
* An ROC curve comparing the trained models saved to ``figures/metabolomics_roc.png``.
* The top 20 RandomForest feature importances saved to
  ``figures/metabolomics_feature_importance.png``.
* A persisted RandomForest model written to ``models/metabolomics_rf.pkl``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split

try:  # pragma: no cover - dependency availability is environment specific.
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully at runtime.
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_AVAILABLE = False

PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("figures")
MODELS_DIR = Path("models")
X_PATH = PROCESSED_DIR / "X_metabolomics.csv"
Y_PATH = PROCESSED_DIR / "y_metabolomics.csv"
ROC_PLOT_PATH = FIGURES_DIR / "metabolomics_roc.png"
IMPORTANCE_PLOT_PATH = FIGURES_DIR / "metabolomics_feature_importance.png"
MODEL_PATH = MODELS_DIR / "metabolomics_rf.pkl"
RANDOM_STATE = 42


def load_processed_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the harmonised metabolomics matrix and aligned binary labels."""

    if not X_PATH.exists() or not Y_PATH.exists():
        raise FileNotFoundError(
            "Processed metabolomics files not found. Run process_metabolomics.py first."
        )

    X = pd.read_csv(X_PATH, index_col="sample_id")
    y = pd.read_csv(Y_PATH, index_col="sample_id")["label"]

    # Ensure the samples are perfectly aligned between X and y.
    common_index = X.index.intersection(y.index)
    if common_index.empty:
        raise RuntimeError("No overlapping samples between features and labels.")

    X = X.loc[common_index]
    y = y.loc[common_index]
    return X, y.astype(int)


def train_models(X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, object], Dict[str, Dict[str, object]], pd.DataFrame, pd.Series]:
    """Split the data, train the models, and collect evaluation metrics."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    trained_models: Dict[str, object] = {}
    metrics: Dict[str, Dict[str, object]] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:, 1]
        else:
            y_scores = model.decision_function(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_scores)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            target_names=["Control", "Alzheimer"],
            zero_division=0,
        )

        metrics[name] = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "classification_report": report,
            "y_scores": y_scores,
        }
        trained_models[name] = model

        print(f"=== {name} ===")
        print(f"Accuracy      : {accuracy:.3f}")
        print(f"ROC AUC       : {roc_auc:.3f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(report)

    return trained_models, metrics, X_test, y_test


def plot_roc_curves(models: Dict[str, object], X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Plot ROC curves for the trained models and save the figure."""

    if not _MATPLOTLIB_AVAILABLE:
        print(
            "Matplotlib is not installed; skipping ROC curve generation. "
            "Install matplotlib to enable plotting."
        )
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    for name, model in models.items():
        RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test,
            ax=ax,
            name=name,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title("Metabolomics ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_PLOT_PATH, dpi=300)
    plt.close()
    print(f"Saved ROC curve to {ROC_PLOT_PATH}")


def plot_feature_importances(model: RandomForestClassifier, feature_names: pd.Index) -> None:
    """Plot and save the top 20 metabolite importances from the RandomForest."""

    if not _MATPLOTLIB_AVAILABLE:
        print(
            "Matplotlib is not installed; skipping feature importance plot. "
            "Install matplotlib to enable plotting."
        )
        return

    importances = model.feature_importances_
    if importances.size == 0:
        print("RandomForest does not expose feature_importances_. Skipping plot.")
        return

    top_k = min(20, importances.size)
    indices = np.argsort(importances)[::-1][:top_k]
    top_features = feature_names[indices]
    top_importances = importances[indices]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    y_positions = np.arange(top_k)
    plt.barh(y_positions, top_importances[::-1], color="steelblue")
    plt.yticks(y_positions, top_features[::-1])
    plt.xlabel("RandomForest feature importance")
    plt.title("Top metabolite features")
    plt.tight_layout()
    plt.savefig(IMPORTANCE_PLOT_PATH, dpi=300)
    plt.close()
    print(f"Saved feature importances to {IMPORTANCE_PLOT_PATH}")


def save_model(model: RandomForestClassifier) -> None:
    """Persist the trained RandomForest model to disk."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved RandomForest model to {MODEL_PATH}")


def main() -> None:
    """Train metabolomics classifiers and generate evaluation artefacts."""

    X, y = load_processed_data()
    trained_models, metrics, X_test, y_test = train_models(X, y)

    plot_roc_curves(trained_models, X_test, y_test)

    rf_model = trained_models["RandomForest"]
    plot_feature_importances(rf_model, X.columns)
    save_model(rf_model)


if __name__ == "__main__":
    main()
