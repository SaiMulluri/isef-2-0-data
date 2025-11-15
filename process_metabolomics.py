"""Create a machine-learning ready metabolomics matrix from ST000046 and ST000047.

This script harmonises the Mayo Clinic plasma (ST000046) and cerebrospinal fluid
(ST000047) untargeted metabolomics datasets from the Metabolomics Workbench.  The
pipeline consolidates all raw text exports into a single sample-by-metabolite
matrix, derives binary Alzheimer labels, applies basic cleaning, and writes the
processed data to ``data/processed`` so it can be consumed by downstream models.

Key processing steps
--------------------
* Prefer structured ``mwtab`` metadata to extract subject cognitive status.  The
  ``SUBJECT_SAMPLE_FACTORS`` section contains "Cognitive Status" for every sample;
  we map ``AD`` samples to label ``1`` and cognitively normal (``CN``) controls to
  label ``0``.  Mild cognitive impairment (``MCI``) samples are explicitly dropped
  because they represent an intermediate clinical stage that we do not model in
  this binary v1 dataset.
* Parse every ``*_file_*.txt`` table for each study.  The Workbench downloads wrap
  the tab-delimited data in ``<html><body><pre>`` tags, so we strip the HTML shell
  and load the tables with :mod:`pandas`.  Each file contributes a unique set of
  metabolites, therefore the feature names are prefixed with the study ID and
  filename to keep them distinct before we concatenate across files.
* Clean the feature matrix by converting intensities to floats, removing
  metabolites with more than 40% missing values, imputing the remaining gaps via
  per-metabolite medians, applying a log1p transform, and finally z-scoring every
  feature.

Run this script from the repository root:

.. code-block:: bash

   python process_metabolomics.py
"""
from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

RAW_DATA_DIR = Path("data/raw")
STUDIES: Tuple[str, ...] = ("ST000046", "ST000047")
PROCESSED_DIR = Path("data/processed")
X_OUTPUT = PROCESSED_DIR / "X_metabolomics.csv"
Y_OUTPUT = PROCESSED_DIR / "y_metabolomics.csv"
MAX_MISSING_FRACTION = 0.4


def normalize_feature_name(name: str) -> str:
    """Return a filesystem-friendly feature identifier."""

    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    return name.strip("_") or "feature"


def strip_html_shell(text: str) -> str:
    """Remove the simple ``<html><body><pre>`` wrapper used by Workbench exports."""

    text = text.replace("<html><body><pre>", "")
    text = text.replace("</pre></body></html>", "")
    return text


def load_labels_from_mwtab(study_id: str) -> Dict[str, str]:
    """Parse cognitive status annotations from the study's mwtab metadata file."""

    mwtab_path = RAW_DATA_DIR / study_id / f"{study_id}_mwtab.txt"
    if not mwtab_path.exists():
        return {}

    text = mwtab_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"\"Sample ID\":\"(?P<sample>[^\"]+)\",\s*\"Factors\":\{[^}]*\"Cognitive Status\":\"(?P<status>[^\"]+)\"",
        re.MULTILINE,
    )
    labels: Dict[str, str] = {}
    for match in pattern.finditer(text):
        sample = match.group("sample").strip()
        status = match.group("status").strip().upper()
        labels[sample] = status
    return labels


def list_measurement_files(study_id: str) -> List[Path]:
    """Return the ordered list of measurement tables for ``study_id``."""

    study_dir = RAW_DATA_DIR / study_id
    # Prefer MW tabular exports if they contain more than just the header row.
    datatable = study_dir / f"{study_id}_datatable.txt"
    if datatable.exists():
        lines = [line for line in datatable.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) > 1:
            return [datatable]
    # Fall back to the HTML wrapped "file" exports which contain the full table.
    files = sorted(study_dir.glob(f"{study_id}_file_*.txt"))
    if not files:
        raise FileNotFoundError(f"No measurement files found for {study_id} in {study_dir}.")
    return files


def parse_measurement_file(file_path: Path, study_id: str) -> pd.DataFrame:
    """Load a single measurement table and return a sample × feature frame.

    The Workbench export includes a second row labelled ``Factors`` that we skip
    because the cognitive status labels are parsed from the mwtab metadata.  Each
    metabolite column is converted to a numeric float and the resulting DataFrame
    is transposed so that samples form the index and metabolites form the columns.
    """

    text = strip_html_shell(file_path.read_text(encoding="utf-8"))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Measurement file {file_path} is empty.")

    header_line = lines[0]
    data_lines = lines[2:] if len(lines) > 1 and lines[1].lower().startswith("factors") else lines[1:]
    csv_text = "\n".join([header_line] + data_lines)
    table = pd.read_csv(StringIO(csv_text), sep="\t", dtype=str)
    table = table.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)

    if {"Metabolite_name", "RefMet_name"}.issubset(table.columns):
        feature_base = table["Metabolite_name"].fillna("unknown").astype(str)
        refmet = table["RefMet_name"].fillna("unknown").astype(str)
        raw_feature_names = feature_base.str.cat(refmet, sep="__")
        feature_ids = (
            f"{study_id}_{file_path.stem}_"
            + raw_feature_names.apply(normalize_feature_name)
        )
        table.insert(0, "feature_id", feature_ids)
    else:
        raise ValueError(f"Expected Metabolite_name/RefMet_name columns in {file_path}.")

    feature_table = table.set_index("feature_id").drop(columns=["Metabolite_name", "RefMet_name"], errors="ignore")
    feature_table = feature_table.apply(pd.to_numeric, errors="coerce")
    feature_table = feature_table.transpose()
    feature_table.index = feature_table.index.astype(str).str.strip()
    feature_table.index.name = "sample_id"
    return feature_table


def assemble_feature_matrix() -> Tuple[pd.DataFrame, pd.Series]:
    """Combine both studies into a unified feature matrix and label vector."""

    feature_frames: List[pd.DataFrame] = []
    label_map: Dict[str, str] = {}

    for study_id in STUDIES:
        study_labels = load_labels_from_mwtab(study_id)
        if not study_labels:
            raise RuntimeError(f"Failed to extract cognitive status labels for {study_id}.")
        for sample, status in study_labels.items():
            existing = label_map.get(sample)
            if existing is not None and existing != status:
                raise RuntimeError(
                    f"Conflicting labels for sample {sample}: {existing!r} vs {status!r}"
                )
            label_map[sample] = status

        for file_path in list_measurement_files(study_id):
            feature_frames.append(parse_measurement_file(file_path, study_id))

    if not feature_frames:
        raise RuntimeError("No measurement tables were successfully parsed.")

    combined = pd.concat(feature_frames, axis=1, join="outer")
    labels = pd.Series(label_map, name="status")

    # Align to the intersection of samples with features and labels.
    common_samples = combined.index.intersection(labels.index)
    combined = combined.loc[common_samples]
    labels = labels.loc[common_samples]

    # Filter to binary classes (AD vs CN) and drop intermediate diagnoses.
    labels = labels.str.upper()
    binary_mask = labels.isin({"AD", "CN"})
    combined = combined.loc[binary_mask]
    labels = labels.loc[binary_mask]

    return combined, labels


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """Convert intensities to floats, impute, log-transform, and z-score features."""

    X_numeric = X.apply(pd.to_numeric, errors="coerce")
    # Drop metabolites with excessive missingness.
    missing_fraction = X_numeric.isna().mean()
    keep_columns = missing_fraction[missing_fraction <= MAX_MISSING_FRACTION].index
    X_numeric = X_numeric.loc[:, keep_columns]

    if X_numeric.empty:
        raise RuntimeError("All features were filtered out due to missingness.")

    # Drop samples that became completely empty after filtering.
    X_numeric = X_numeric.dropna(axis=0, how="all")

    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(X_numeric)

    log_transformed = np.log1p(imputed)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(log_transformed)

    cleaned = pd.DataFrame(scaled, index=X_numeric.index, columns=X_numeric.columns)
    cleaned.index.name = "sample_id"
    return cleaned


def save_outputs(X: pd.DataFrame, y: pd.Series) -> None:
    """Persist the processed feature matrix and labels to ``data/processed``."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X.to_csv(X_OUTPUT, index=True, index_label="sample_id")
    y.to_frame(name="label").to_csv(Y_OUTPUT, index=True, index_label="sample_id")
    print(f"Saved features to {X_OUTPUT}")
    print(f"Saved labels to {Y_OUTPUT}")


def main() -> None:
    """Execute the processing pipeline for the Mayo metabolomics studies."""

    features, status = assemble_feature_matrix()
    print(f"Loaded {features.shape[0]} samples and {features.shape[1]} raw metabolites.")

    cleaned_features = clean_features(features)
    aligned_status = status.loc[cleaned_features.index]
    binary_labels = aligned_status.map({"AD": 1, "CN": 0}).astype(int)

    print(
        "Final dataset:"
        f" {cleaned_features.shape[0]} samples × {cleaned_features.shape[1]} features"
    )
    print("Class balance:")
    print(binary_labels.value_counts().rename({1: "AD", 0: "CN"}))

    save_outputs(cleaned_features, binary_labels)


if __name__ == "__main__":
    main()
