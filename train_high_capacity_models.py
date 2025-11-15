"""Train scalable models with leave-one-study-out evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


def load_datasets(processed_dir: Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    X_path = processed_dir / "X_all_expanded.csv"
    y_path = processed_dir / "y_all_expanded.csv"
    study_path = processed_dir / "study_all_expanded.csv"

    if not (X_path.exists() and y_path.exists() and study_path.exists()):
        raise FileNotFoundError(
            "Processed data not found. Run `python -m preprocessing.multi_dataset_processor` first."
        )

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).iloc[:, 0]
    studies = pd.read_csv(study_path).iloc[:, 0]
    return X, y, studies


def get_model_builders(random_state: int = 42) -> Dict[str, Callable[[], object]]:
    return {
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced_subsample",
        ),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
            max_depth=7,
            learning_rate=0.1,
            max_leaf_nodes=31,
            random_state=random_state,
        ),
        "LogisticRegression": lambda: Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="lbfgs",
                        penalty="l2",
                        C=1.0,
                        max_iter=1000,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }


def maybe_add_optional_models(builders: Dict[str, Callable[[], object]]) -> Dict[str, Callable[[], object]]:
    try:
        from xgboost import XGBClassifier

        builders["XGBoost"] = lambda: XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            reg_lambda=1.0,
            random_state=42,
        )
    except Exception:  # pragma: no cover - optional dependency
        logger.info("XGBoost not available; skipping.")

    try:
        from lightgbm import LGBMClassifier

        builders["LightGBM"] = lambda: LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    except Exception:  # pragma: no cover - optional dependency
        logger.info("LightGBM not available; skipping.")

    return builders


def tune_or_fit_model(model_name: str, estimator, X_train: pd.DataFrame, y_train: pd.Series):
    if model_name == "RandomForest":
        param_distributions = {
            "n_estimators": [200, 300, 400, 500],
            "max_depth": [None, 5, 10, 15],
            "min_samples_leaf": [1, 2, 4],
        }
        search = RandomizedSearchCV(
            estimator,
            param_distributions=param_distributions,
            n_iter=5,
            cv=3,
            scoring="roc_auc",
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_params_

    estimator.fit(X_train, y_train)
    return estimator, {}


def evaluate_model(estimator, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_pred = estimator.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    try:
        if hasattr(estimator, "predict_proba"):
            y_prob = estimator.predict_proba(X_test)[:, 1]
        elif hasattr(estimator, "decision_function"):
            y_prob = estimator.decision_function(X_test)
        else:
            y_prob = None
        if y_prob is not None and len(np.unique(y_test)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
        else:
            metrics["roc_auc"] = float("nan")
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to compute ROC AUC: %s", exc)
        metrics["roc_auc"] = float("nan")

    cm = confusion_matrix(y_test, y_pred)
    metrics["tn"] = float(cm[0, 0]) if cm.shape == (2, 2) else float("nan")
    metrics["fp"] = float(cm[0, 1]) if cm.shape == (2, 2) else float("nan")
    metrics["fn"] = float(cm[1, 0]) if cm.shape == (2, 2) else float("nan")
    metrics["tp"] = float(cm[1, 1]) if cm.shape == (2, 2) else float("nan")
    return metrics


def leave_one_study_out(
    X: pd.DataFrame, y: pd.Series, studies: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    unique_studies = sorted(studies.unique())
    model_builders = maybe_add_optional_models(get_model_builders())

    per_study_records: List[Dict[str, object]] = []
    best_params: Dict[str, Dict[str, float]] = {}

    for holdout_study in unique_studies:
        logger.info("Evaluating hold-out study %s", holdout_study)
        mask = studies == holdout_study
        X_train, X_test = X.loc[~mask], X.loc[mask]
        y_train, y_test = y.loc[~mask], y.loc[mask]

        if X_test.empty or X_train.empty:
            logger.warning("Insufficient data for study %s; skipping.", holdout_study)
            continue

        for model_name, builder in model_builders.items():
            estimator = builder()
            tuned_estimator, params = tune_or_fit_model(model_name, estimator, X_train, y_train)
            metrics = evaluate_model(tuned_estimator, X_test, y_test)
            record = {
                "study_id": holdout_study,
                "model": model_name,
                **metrics,
            }
            per_study_records.append(record)
            if params:
                best_params.setdefault(model_name, {}).update({holdout_study: params})

    if not per_study_records:
        raise RuntimeError("No evaluation records produced during LOSO.")

    per_study_df = pd.DataFrame(per_study_records)

    summary: Dict[str, Dict[str, float]] = {}
    for model_name, group in per_study_df.groupby("model"):
        summary[model_name] = {
            "mean_accuracy": float(group["accuracy"].mean()),
            "std_accuracy": float(group["accuracy"].std(ddof=0)),
            "mean_roc_auc": float(group["roc_auc"].mean(skipna=True)),
            "std_roc_auc": float(group["roc_auc"].std(ddof=0, skipna=True)),
        }

    return per_study_df, summary, best_params


def select_best_model(summary: Dict[str, Dict[str, float]]) -> str:
    best_model = None
    best_score = -np.inf
    for model_name, metrics in summary.items():
        score = metrics.get("mean_roc_auc", float("nan"))
        if np.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_model = model_name
    if best_model is None:
        # fallback to accuracy if ROC AUC was undefined
        for model_name, metrics in summary.items():
            score = metrics.get("mean_accuracy", float("nan"))
            if np.isnan(score):
                continue
            if score > best_score:
                best_score = score
                best_model = model_name
    if best_model is None:
        raise RuntimeError("Unable to select a best model based on evaluation metrics.")
    return best_model


def retrain_best_model(
    model_name: str, X: pd.DataFrame, y: pd.Series, output_path: Path
) -> None:
    builders = maybe_add_optional_models(get_model_builders())
    if model_name not in builders:
        raise KeyError(f"Model {model_name} not recognized.")
    estimator = builders[model_name]()
    estimator.fit(X, y)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, output_path)
    logger.info("Saved final %s model to %s", model_name, output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    processed_dir = Path("data/processed")
    X, y, studies = load_datasets(processed_dir)

    per_study_df, summary, _ = leave_one_study_out(X, y, studies)

    per_study_path = processed_dir / "high_capacity_eval_by_study.csv"
    summary_path = processed_dir / "high_capacity_eval_summary.json"

    per_study_df.to_csv(per_study_path, index=False)
    with summary_path.open("w") as fp:
        json.dump(summary, fp, indent=2)

    logger.info("Saved per-study evaluation to %s", per_study_path)
    logger.info("Saved summary metrics to %s", summary_path)

    best_model_name = select_best_model(summary)
    logger.info("Best model selected: %s", best_model_name)

    model_output_path = Path("models/high_capacity_metabolomics_model.pkl")
    retrain_best_model(best_model_name, X, y, model_output_path)


if __name__ == "__main__":
    main()

