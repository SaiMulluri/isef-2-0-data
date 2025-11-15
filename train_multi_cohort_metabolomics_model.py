"""Train multi-cohort Alzheimer classifiers with leave-one-study-out validation.

The harmonised metabolomics matrices produced by ``process_multi_metabolomics``
combine five Metabolomics Workbench studies (ST000046, ST000047, ST000462,
ST001152, ST001050).  Each sample is annotated with binary Alzheimer (1) versus
cognitively normal control (0) labels plus engineered pathway scores for the
tryptophan (TRP) and short-chain fatty acid (SCFA) pathways.  This script trains
scikit-learn models while holding out one study at a time to quantify
cross-cohort generalisation.

Workflow
========
1. Load ``X_all_metabolomics.csv``, ``y_all_metabolomics.csv`` and
   ``study_all_metabolomics.csv``.
2. For every study, treat it as a test fold and train on the remaining cohorts
   (leave-one-study-out CV).  We report accuracy, ROC AUC and confusion matrices.
3. Fit a ``RandomForestClassifier`` (n_estimators=300, class_weight="balanced")
   on each fold.  A logistic regression baseline is also attempted when feasible
   to provide a linear reference model.
4. After cross-validation, fit the RandomForest on the entire dataset to extract
   feature importances, save them to ``data/processed/metabolomics_feature_importance.csv``
   and optionally plot the top 20 features (if matplotlib is installed).
5. Persist the all-data RandomForest model to ``models/multi_study_metabolomics_rf.pkl``.

Run from the repository root::

   python train_multi_cohort_metabolomics_model.py
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

FIGURES_DIR = Path("figures")
MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")


def load_data() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load harmonised feature matrix, labels, and study identifiers."""

    X_path = PROCESSED_DIR / "X_all_metabolomics.csv"
    y_path = PROCESSED_DIR / "y_all_metabolomics.csv"
    study_path = PROCESSED_DIR / "study_all_metabolomics.csv"

    if not X_path.exists():
        raise FileNotFoundError(f"Feature matrix not found at {X_path}. Run process script first.")

    X = pd.read_csv(X_path, index_col="sample_id")
    y = pd.read_csv(y_path, index_col=0).iloc[:, 0]
    study = pd.read_csv(study_path, index_col=0).iloc[:, 0]

    return X, y.astype(int), study.astype(str)


def ensure_directories() -> None:
    """Ensure output folders exist."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def leave_one_study_out(
    X: pd.DataFrame, y: pd.Series, study: pd.Series
) -> Tuple[List[Dict[str, float]], List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    """Run leave-one-study-out CV and collect metrics and predictions."""

    metrics: List[Dict[str, float]] = []
    confusion_matrices: List[Dict[str, np.ndarray]] = []
    predictions: List[Dict[str, np.ndarray]] = []

    unique_studies = sorted(study.unique())
    logging.info("Leave-one-study-out CV across cohorts: %s", ", ".join(unique_studies))

    for holdout in unique_studies:
        train_mask = study != holdout
        test_mask = study == holdout

        X_train, X_test = X.loc[train_mask], X.loc[test_mask]
        y_train, y_test = y.loc[train_mask], y.loc[test_mask]

        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        metrics.append({"study": holdout, "accuracy": acc, "roc_auc": auc})
        confusion_matrices.append({"study": holdout, "matrix": cm})
        predictions.append({"study": holdout, "y_true": y_test.values, "y_score": y_proba})

        logging.info(
            "Hold-out %s -> accuracy %.3f, ROC AUC %.3f, confusion matrix %s",
            holdout,
            acc,
            auc,
            cm.tolist(),
        )

        try:
            lr = LogisticRegression(
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                max_iter=200,
            )
            lr.fit(X_train, y_train)
            lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
            logging.info("  LogisticRegression ROC AUC on %s fold: %.3f", holdout, lr_auc)
        except Exception as exc:  # noqa: BLE001 - log but keep pipeline running
            logging.warning("  LogisticRegression failed on %s fold: %s", holdout, exc)

    return metrics, confusion_matrices, predictions


def aggregate_metrics(metrics: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    """Compute mean and std for accuracy and ROC AUC across folds."""

    accuracies = np.array([m["accuracy"] for m in metrics])
    aucs = np.array([m["roc_auc"] for m in metrics])
    return {
        "accuracy": (float(np.mean(accuracies)), float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0),
        "roc_auc": (float(np.mean(aucs)), float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0),
    }


def plot_feature_importance(features: List[str], importances: np.ndarray) -> None:
    """Plot top 20 feature importances if matplotlib is available."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency in sandbox
        logging.warning("matplotlib not available; skipping feature importance plot")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    order = np.argsort(importances)[::-1][:20]
    top_features = [features[i] for i in order]
    top_importances = importances[order]

    plt.figure(figsize=(8, 6))
    plt.barh(top_features[::-1], top_importances[::-1])
    plt.xlabel("RandomForest importance")
    plt.title("Top metabolite and pathway features")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "metabolomics_feature_importance.png", dpi=200)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, label: str) -> None:
    """Plot ROC curve for a representative fold."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        logging.warning("matplotlib not available; skipping ROC curve plot")
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc_score(y_true, y_score):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve leave-out {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "metabolomics_multi_study_roc.png", dpi=200)
    plt.close()


def save_feature_importances(clf: RandomForestClassifier, feature_names: List[str]) -> None:
    """Save feature importances to CSV and optionally plot them."""

    importances = clf.feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_path = PROCESSED_DIR / "metabolomics_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    logging.info("Saved feature importances to %s", importance_path)

    plot_feature_importance(feature_names, importances)


def main() -> None:
    """Execute the multi-cohort training workflow."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ensure_directories()

    X, y, study = load_data()

    metrics, confusion_matrices, predictions = leave_one_study_out(X, y, study)
    agg = aggregate_metrics(metrics)
    logging.info(
        "Mean accuracy %.3f ± %.3f, mean ROC AUC %.3f ± %.3f",
        agg["accuracy"][0],
        agg["accuracy"][1],
        agg["roc_auc"][0],
        agg["roc_auc"][1],
    )

    # Plot ROC for the first fold as a representative example
    if predictions:
        representative = predictions[0]
        plot_roc_curve(representative["y_true"], representative["y_score"], representative["study"])

    # Fit on all data for interpretability and export artefacts
    full_clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    full_clf.fit(X, y)
    save_feature_importances(full_clf, list(X.columns))

    model_path = MODELS_DIR / "multi_study_metabolomics_rf.pkl"
    joblib.dump(full_clf, model_path)
    logging.info("Saved RandomForest model to %s", model_path)


if __name__ == "__main__":
    main()
