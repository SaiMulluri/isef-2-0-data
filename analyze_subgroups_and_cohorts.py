"""Evaluate trained models across cohorts and demographic subgroups."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


logger = logging.getLogger(__name__)


def load_processed_outputs(processed_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = pd.read_csv(processed_dir / "X_all_expanded.csv")
    y = pd.read_csv(processed_dir / "y_all_expanded.csv").iloc[:, 0]
    studies = pd.read_csv(processed_dir / "study_all_expanded.csv").iloc[:, 0]
    return X, y, studies


def compute_binary_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    # Sensitivity and specificity
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    metrics["sensitivity"] = float(sensitivity)
    metrics["specificity"] = float(specificity)
    metrics["n_samples"] = int(len(y_true))
    return metrics


def evaluate_subgroups(
    X: pd.DataFrame,
    y: pd.Series,
    studies: pd.Series,
    model,
) -> pd.DataFrame:
    y_pred = model.predict(X)
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X)[:, 1]
        except Exception:  # pragma: no cover - fallback
            y_prob = None
    elif hasattr(model, "decision_function"):
        try:
            y_prob = model.decision_function(X)
        except Exception:  # pragma: no cover - fallback
            y_prob = None

    records: List[Dict[str, object]] = []

    # Cohort-level evaluation
    for study_id in sorted(studies.unique()):
        mask = studies == study_id
        metrics = compute_binary_metrics(y[mask], y_pred[mask], y_prob[mask] if y_prob is not None else None)
        metrics.update({"group_type": "cohort", "group_name": study_id})
        records.append(metrics)

    # Age-based subgroups
    if "age_years" in X.columns:
        age_values = X["age_years"]
        if age_values.notna().any():
            younger_mask = age_values < 70
            older_mask = age_values >= 70
            for mask, name in [(younger_mask, "age_lt_70"), (older_mask, "age_gte_70")]:
                if mask.sum() == 0:
                    continue
                metrics = compute_binary_metrics(
                    y[mask], y_pred[mask], y_prob[mask] if y_prob is not None else None
                )
                metrics.update({"group_type": "subgroup", "group_name": name})
                records.append(metrics)

    # Sex-based subgroups
    sex_columns = [col for col in X.columns if col.lower().startswith("sex_")]
    if sex_columns:
        for sex_col in sex_columns:
            mask = X[sex_col] == 1
            if mask.sum() == 0:
                continue
            metrics = compute_binary_metrics(
                y[mask], y_pred[mask], y_prob[mask] if y_prob is not None else None
            )
            metrics.update({"group_type": "subgroup", "group_name": sex_col})
            records.append(metrics)

    if not records:
        raise RuntimeError("No subgroup or cohort evaluations were generated.")

    return pd.DataFrame(records)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    processed_dir = Path("data/processed")
    model_path = Path("models/high_capacity_metabolomics_model.pkl")

    if not model_path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python train_high_capacity_models.py` first."
        )

    X, y, studies = load_processed_outputs(processed_dir)
    model = joblib.load(model_path)

    results_df = evaluate_subgroups(X, y, studies, model)
    output_path = processed_dir / "subgroup_and_cohort_performance.csv"
    results_df.to_csv(output_path, index=False)
    logger.info("Saved subgroup and cohort performance to %s", output_path)


if __name__ == "__main__":
    main()

