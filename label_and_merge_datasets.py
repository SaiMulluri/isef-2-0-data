"""Label and merge Metabolomics Workbench datasets into ML-ready CSV files."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


DATA_DIR = Path("data")
MW_PATTERN = DATA_DIR / "metabolomics_workbench" / "*" / "*_datatable.txt"
PROCESSED_DIR = DATA_DIR / "processed"
X_OUTPUT = PROCESSED_DIR / "X_metabolomics.csv"
Y_OUTPUT = PROCESSED_DIR / "y_metabolomics.csv"

LABEL_KEYWORDS = {
    "alzheim": 1,
    "control": 0,
    "healthy": 0,
    "normal": 0,
}

SAMPLE_KEY_CANDIDATES = ["sample", "subject", "patient", "participant"]
PHENOTYPE_KEY_CANDIDATES = ["disease", "diagnosis", "phenotype", "condition", "group"]


def find_column(columns: List[str], keywords: List[str]) -> Optional[str]:
    """Return the first column name containing any keyword (case-insensitive)."""
    lowered = [c.lower() for c in columns]
    for keyword in keywords:
        for idx, col_lower in enumerate(lowered):
            if keyword in col_lower:
                return columns[idx]
    return None


def infer_label(value: str) -> Optional[int]:
    value_lower = value.lower()
    for keyword, label in LABEL_KEYWORDS.items():
        if keyword in value_lower:
            return label
    return None


def normalize_feature_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    return name.strip("_") or "feature"


def process_single_file(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    except Exception as exc:
        print(f"Failed to read {path}: {exc}")
        return None

    if df.empty:
        print(f"Warning: {path} is empty.")
        return None

    columns = list(df.columns)
    sample_col = find_column(columns, SAMPLE_KEY_CANDIDATES)
    phenotype_col = find_column(columns, PHENOTYPE_KEY_CANDIDATES)

    if phenotype_col is None:
        print(f"Skipping {path}: phenotype column not found.")
        return None

    if sample_col is None:
        # Create a synthetic sample ID column using the index.
        sample_col = "__sample_id__"
        df[sample_col] = [f"{path.stem}_{i}" for i in range(len(df))]

    labels: List[Optional[int]] = []
    for val in df[phenotype_col].fillna(""):
        label = infer_label(str(val)) if val is not None else None
        labels.append(label)

    df["label"] = labels
    df = df.dropna(subset=["label"])
    if df.empty:
        print(f"Skipping {path}: no determinable labels.")
        return None

    df["label"] = df["label"].astype(int)

    # Convert feature columns to numeric where possible.
    exclude_cols = {sample_col, phenotype_col, "label"}
    feature_frames: Dict[str, pd.Series] = {}
    for col in columns:
        if col in exclude_cols:
            continue
        normalized_name = normalize_feature_name(col)
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            feature_frames[normalized_name] = series

    if not feature_frames:
        print(f"Skipping {path}: no numeric features identified.")
        return None

    features_df = pd.DataFrame(feature_frames)
    features_df.insert(0, "sample_id", df[sample_col].astype(str).fillna(""))
    features_df["label"] = df["label"].values
    return features_df


def merge_datasets() -> Tuple[pd.DataFrame, pd.Series]:
    processed_frames: List[pd.DataFrame] = []
    for path in sorted(Path().glob(str(MW_PATTERN))):
        print(f"Processing {path}")
        frame = process_single_file(path)
        if frame is not None:
            processed_frames.append(frame)

    if not processed_frames:
        raise RuntimeError(
            "No valid metabolomics datasets found. Ensure datatable files are downloaded."
        )

    combined = pd.concat(processed_frames, axis=0, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["sample_id"])

    labels = combined.pop("label")
    combined = combined.drop(columns=["sample_id"], errors="ignore")

    # Drop columns that are mostly missing.
    missing_fraction = combined.isna().mean()
    keep_columns = missing_fraction[missing_fraction <= 0.8].index
    combined = combined[keep_columns]

    # Impute remaining missing values with column medians.
    combined = combined.apply(pd.to_numeric, errors="coerce")
    medians = combined.median()
    combined = combined.fillna(medians)

    return combined, labels


def save_outputs(X: pd.DataFrame, y: pd.Series) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X.to_csv(X_OUTPUT, index=False)
    y.to_csv(Y_OUTPUT, index=False, header=["label"])
    print("Saved", X_OUTPUT)
    print("Saved", Y_OUTPUT)


def report(X: pd.DataFrame, y: pd.Series) -> None:
    n_samples = len(y)
    n_features = X.shape[1]
    class_counts = y.value_counts().to_dict()
    print("=== Metabolomics Dataset Summary ===")
    print(f"Samples: {n_samples}")
    print(f"Features: {n_features}")
    print("Class balance:")
    for label, count in sorted(class_counts.items()):
        label_name = "Alzheimer" if label == 1 else "Control"
        print(f"  {label_name} ({label}): {count}")


def main() -> None:
    X, y = merge_datasets()
    save_outputs(X, y)
    report(X, y)


if __name__ == "__main__":
    main()
