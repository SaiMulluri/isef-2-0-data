"""Train machine learning models for Alzheimer detection using metabolomics data."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier  # type: ignore

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    XGBOOST_AVAILABLE = False


DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
X_PATH = DATA_DIR / "X_metabolomics.csv"
Y_PATH = DATA_DIR / "y_metabolomics.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
REPORT_PATH = MODELS_DIR / "best_model_report.txt"


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    if not X_PATH.exists() or not Y_PATH.exists():
        raise FileNotFoundError(
            "Processed datasets not found. Please run label_and_merge_datasets.py first."
        )
    X = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH)["label"]
    return X, y


def prepare_data(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train.to_numpy(), y_test.to_numpy(), scaler


def get_models() -> Dict[str, object]:
    models: Dict[str, object] = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVC": SVC(probability=True, random_state=42),
        "MLPClassifier": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBClassifier"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            use_label_encoder=False,
        )
    else:
        print("xgboost not installed; skipping XGBClassifier.")
    return models


def evaluate_model(model, X_train, X_test, y_train, y_test) -> Dict[str, float | List[List[int]]]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
        y_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-8)
    else:
        y_proba = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc_auc = float("nan")
    cm = confusion_matrix(y_test, y_pred)
    return {
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
    }


def train_and_evaluate() -> Tuple[str, object, StandardScaler, Dict[str, Dict[str, float | List[List[int]]]]]:
    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models = get_models()
    results: Dict[str, Dict[str, float | List[List[int]]]] = {}

    best_model_name = None
    best_model = None
    best_score = -float("inf")

    print("=== Model Performance ===")
    print(f"{'Model':<20} {'Accuracy':>10} {'ROC-AUC':>10}")
    for name, model in models.items():
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[name] = metrics
        roc_auc = metrics.get("roc_auc", float("nan"))
        accuracy = metrics.get("accuracy", float("nan"))
        print(f"{name:<20} {accuracy:>10.4f} {roc_auc:>10.4f}")
        if np.isnan(roc_auc):
            score = accuracy
        else:
            score = roc_auc
        if score > best_score:
            best_score = score
            best_model_name = name
            best_model = model

    print("\nConfusion Matrices:")
    for name, metrics in results.items():
        cm = metrics["confusion_matrix"]
        print(f"{name}: {cm}")

    if best_model_name is None or best_model is None:
        raise RuntimeError("No model was successfully trained.")

    return best_model_name, best_model, scaler, results[best_model_name]


def save_artifacts(model_name: str, model, scaler: StandardScaler, metrics: Dict[str, float | List[List[int]]]) -> None:
    joblib.dump(model, BEST_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    report_lines = [
        f"Best model: {model_name}",
        f"Test accuracy: {metrics.get('accuracy', float('nan')):.4f}",
        f"Test ROC-AUC: {metrics.get('roc_auc', float('nan')):.4f}",
        f"Source data: {X_PATH.name}, {Y_PATH.name}",
    ]
    REPORT_PATH.write_text("\n".join(report_lines) + "\n")
    print(f"Saved best model to {BEST_MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")
    print(f"Wrote report to {REPORT_PATH}")


def main() -> None:
    model_name, model, scaler, metrics = train_and_evaluate()
    save_artifacts(model_name, model, scaler, metrics)


if __name__ == "__main__":
    main()
