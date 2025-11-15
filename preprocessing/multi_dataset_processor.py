"""Multi-cohort metabolomics preprocessing for the ISEF 2.0 Alzheimer's project.

This module builds a unified feature matrix across multiple Metabolomics Workbench
studies. For each registered human metabolomics cohort, it:

- Loads raw tabular data from data/raw/<ST_ID>/.
- Extracts samples, metabolite intensities, and diagnosis labels.
- Maps textual diagnoses to binary labels (1 = AD-like, 0 = control-like).
- Harmonizes metabolite names for key tryptophan (TRP) and short-chain fatty
  acid (SCFA) pathway metabolites.
- Applies quality control (missingness filtering), imputation, log1p transform,
  and per-study z-scoring.
- Derives pathway-level scores:
    * TRP_pathway_score
    * SCFA_pathway_score
- Adds a one-hot indicator for each study.

It then concatenates all processed cohorts into:

    data/processed/X_all_expanded.csv
    data/processed/y_all_expanded.csv
    data/processed/study_all_expanded.csv

This pipeline is intended for RESEARCH AND EDUCATIONAL USE ONLY. It has not been
validated as a clinical diagnostic tool and must not be used to diagnose, screen,
or treat any patient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and dataset registry
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Minimal initial registry; you can add more ST IDs here later.
DATASET_REGISTRY: Dict[str, Dict[str, str]] = {
    "ST000046": {
        "dataset_id": "ST000046",
        "species": "human",
        "data_type": "metabolomics",
        "matrix": "plasma",
        "raw_dir": str(RAW_DIR / "ST000046"),
        "notes": "Metabolomics Workbench ST000046 (human plasma, AD/MCI/CN).",
    },
    "ST000047": {
        "dataset_id": "ST000047",
        "species": "human",
        "data_type": "metabolomics",
        "matrix": "CSF",
        "raw_dir": str(RAW_DIR / "ST000047"),
        "notes": "Metabolomics Workbench ST000047 (human CSF, AD/MCI/CN).",
    },
    # Add additional cohorts (ST000462, ST001152, ST001050, etc.) here as needed.
}

# ---------------------------------------------------------------------------
# Metabolite synonym maps and pathway definitions
# ---------------------------------------------------------------------------

TRP_SYNONYMS = {
    "tryptophan": "Tryptophan",
    "l-tryptophan": "Tryptophan",
    "indole-3-propionic acid": "Indole-3-propionic acid",
    "indolepropionic acid": "Indole-3-propionic acid",
    "indole-3-acetic acid": "Indole-3-acetic acid",
    "indoleacetic acid": "Indole-3-acetic acid",
    "indole-3-lactic acid": "Indole-3-lactic acid",
    "indolelactic acid": "Indole-3-lactic acid",
    "kynurenine": "Kynurenine",
    "kynurenic acid": "Kynurenic acid",
    "quinolinic acid": "Quinolinic acid",
    "3-hydroxykynurenine": "3-Hydroxykynurenine",
    "tryptamine": "Tryptamine",
}

SCFA_SYNONYMS = {
    "acetate": "Acetate",
    "acetic acid": "Acetate",
    "propionate": "Propionate",
    "propionic acid": "Propionate",
    "butyrate": "Butyrate",
    "butyric acid": "Butyrate",
    "valerate": "Valerate",
    "valeric acid": "Valerate",
    "isobutyrate": "Isobutyrate",
    "isobutyric acid": "Isobutyrate",
}

TRP_METABS: List[str] = [
    "Tryptophan",
    "Indole-3-propionic acid",
    "Indole-3-acetic acid",
    "Indole-3-lactic acid",
    "Kynurenine",
    "Kynurenic acid",
    "Quinolinic acid",
    "3-Hydroxykynurenine",
    "Tryptamine",
]

SCFA_METABS: List[str] = [
    "Acetate",
    "Propionate",
    "Butyrate",
    "Valerate",
    "Isobutyrate",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """Find the first column in df whose name matches one of candidates (case-insensitive).

    Raises a ValueError if no candidate is found.
    """
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for col_lower, col_orig in lower_map.items():
            if cand.lower() == col_lower:
                return col_orig
    # also allow substring matches
    for cand in candidates:
        for col in df.columns:
            if cand.lower() in col.lower():
                return col
    raise ValueError(f"None of the candidate columns {candidates} were found in: {list(df.columns)}")


def harmonize_metabolite_names(columns: pd.Index) -> pd.Index:
    """Map known TRP/SCFA metabolites to canonical names.

    Uses TRP_SYNONYMS and SCFA_SYNONYMS (case-insensitive). Other names are left unchanged.
    """
    new_names: List[str] = []
    for col in columns:
        key = col.lower().strip()
        if key in TRP_SYNONYMS:
            new_names.append(TRP_SYNONYMS[key])
        elif key in SCFA_SYNONYMS:
            new_names.append(SCFA_SYNONYMS[key])
        else:
            new_names.append(col)
    return pd.Index(new_names)


def load_study_raw(study_cfg: Dict[str, str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and parse raw Metabolomics Workbench data for a single study.

    Parameters
    ----------
    study_cfg : dict
        Configuration dict from DATASET_REGISTRY for this study.

    Returns
    -------
    X_raw : DataFrame
        Wide-format matrix with rows = samples and columns = metabolite features
        plus any included metadata.
    y_raw : Series
        Textual diagnosis labels aligned with X_raw.index.

    Notes
    -----
    - This function is designed for MW-style tabular exports.
    - It uses heuristic column-name detection for sample ID and diagnosis.
    - It may need minor adjustments per study if formats differ slightly.
    """
    raw_dir = Path(study_cfg["raw_dir"])
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found for study {study_cfg['dataset_id']}: {raw_dir}")
    # pick first txt or csv file
    files = list(raw_dir.glob("*.txt")) + list(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No .txt or .csv files found in {raw_dir}")
    data_path = files[0]

    if data_path.suffix == ".txt":
        df = pd.read_csv(data_path, sep="\t", dtype=str)
    else:
        df = pd.read_csv(data_path, dtype=str)

    # heuristic column detection
    sample_col = _find_column(df, ["Sample", "Sample_ID", "SampleID", "Subject_ID", "SubjectID", "ID"])
    diag_col = _find_column(df, ["Diagnosis", "Group", "Condition", "Disease", "dx"])

    df = df.set_index(sample_col)
    y_raw = df[diag_col].copy()
    X_raw = df.drop(columns=[diag_col])

    return X_raw, y_raw


def map_diagnosis_to_binary(y_raw: pd.Series) -> pd.Series:
    """Map textual diagnosis labels to binary AD vs control labels.

    Mapping
    -------
    - 1 = Alzheimer's disease / probable AD (labels containing 'alzheimer', 'alzheimers', or exactly 'ad')
    - 0 = cognitively normal controls (labels containing 'control', 'cn', 'normal', 'healthy')
    - Other labels (e.g., 'mci', 'mild cognitive impairment', 'other') are dropped.

    Returns
    -------
    y_bin : Series
        Series of 0/1 labels indexed like the input, with ambiguous labels removed.
    """
    y = y_raw.astype(str).str.lower().str.strip()

    def _map_one(val: str):
        if "alzheimer" in val or val == "ad" or val == "alzheimers":
            return 1
        if "control" in val or val == "cn" or "normal" in val or "healthy" in val:
            return 0
        return np.nan

    mapped = y.apply(_map_one)
    y_bin = mapped.dropna().astype(int)
    return y_bin


def clean_and_transform_single_dataset(
    X_raw: pd.DataFrame, y_bin: pd.Series, study_id: str
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Clean and transform a single study's data.

    Steps
    -----
    - Align X_raw and y_bin by sample index.
    - Convert numeric columns to float.
    - Harmonize metabolite names (TRP/SCFA).
    - Drop columns with high missingness (>50%).
    - Impute remaining missing values with per-feature median.
    - Apply log1p transform to all intensity features.
    - Z-score features within this study.
    - Compute TRP_pathway_score and SCFA_pathway_score if enough metabolites are present.
    - Add a one-hot indicator column for this study: f"study_{study_id}" = 1.0 for all samples.

    Returns
    -------
    X_proc : DataFrame
    y_proc : Series
    study_series : Series
        A Series where all entries are the string study_id.
    """
    # align indices
    common_idx = X_raw.index.intersection(y_bin.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping samples between features and labels for study {study_id}")
    X = X_raw.loc[common_idx].copy()
    y = y_bin.loc[common_idx].copy()

    # keep only numeric columns
    numeric_cols = []
    for col in X.columns:
        try:
            pd.to_numeric(X[col].dropna().iloc[:5])  # small sample test
            numeric_cols.append(col)
        except Exception:
            continue
    X = X[numeric_cols]

    # harmonize names
    X.columns = harmonize_metabolite_names(X.columns)

    # convert to float and compute missingness
    X = X.apply(pd.to_numeric, errors="coerce")
    missing_frac = X.isna().mean()
    keep_cols = missing_frac[missing_frac <= 0.5].index  # drop columns with >50% missing
    X = X[keep_cols]

    # impute median
    X = X.fillna(X.median())

    # log1p and z-score
    values = X.values.astype(float)
    values = np.log1p(values)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1.0
    values = (values - means) / stds
    X = pd.DataFrame(values, index=X.index, columns=X.columns)

    # pathway scores
    present_trp = [m for m in TRP_METABS if m in X.columns]
    if len(present_trp) >= 2:
        X["TRP_pathway_score"] = X[present_trp].mean(axis=1)

    present_scfa = [m for m in SCFA_METABS if m in X.columns]
    if len(present_scfa) >= 2:
        X["SCFA_pathway_score"] = X[present_scfa].mean(axis=1)

    # study indicator
    X[f"study_{study_id}"] = 1.0

    study_series = pd.Series(study_id, index=X.index, name="study_id")

    return X, y, study_series


def build_multi_dataset_features() -> None:
    """Process all human metabolomics datasets and build combined feature tables.

    Loops over all entries in DATASET_REGISTRY where:

        species == "human" and data_type == "metabolomics"

    For each study, it attempts to load raw data, map diagnoses to 0/1 labels,
    clean and normalize features, derive TRP/SCFA pathway scores, and add a
    study indicator. It then concatenates all processed cohorts and writes:

        data/processed/X_all_expanded.csv
        data/processed/y_all_expanded.csv
        data/processed/study_all_expanded.csv
    """
    human_metabos = {
        ds_id: cfg
        for ds_id, cfg in DATASET_REGISTRY.items()
        if cfg.get("species") == "human" and cfg.get("data_type") == "metabolomics"
    }

    X_list: List[pd.DataFrame] = []
    y_list: List[pd.Series] = []
    study_list: List[pd.Series] = []

    for study_id, cfg in human_metabos.items():
        raw_dir = Path(cfg["raw_dir"])
        if not raw_dir.exists():
            print(f"[WARN] Raw directory missing for {study_id}: {raw_dir}; skipping.")
            continue
        if not any(raw_dir.glob("*.txt")) and not any(raw_dir.glob("*.csv")):
            print(f"[WARN] No .txt/.csv files found for {study_id} in {raw_dir}; skipping.")
            continue

        print(f"[INFO] Processing study {study_id} from {raw_dir}...")

        try:
            X_raw, y_raw = load_study_raw(cfg)
        except Exception as exc:
            print(f"[WARN] Failed to load raw data for {study_id}: {exc}; skipping.")
            continue

        y_bin = map_diagnosis_to_binary(y_raw)
        if y_bin.empty:
            print(f"[WARN] No usable AD vs Control labels for {study_id}; skipping.")
            continue

        try:
            X_proc, y_proc, study_series = clean_and_transform_single_dataset(X_raw, y_bin, study_id)
        except Exception as exc:
            print(f"[WARN] Failed to process study {study_id}: {exc}; skipping.")
            continue

        X_list.append(X_proc)
        y_list.append(y_proc)
        study_list.append(study_series)

    if not X_list:
        raise RuntimeError("No datasets were successfully processed. Check raw data and registry.")

    X_all = pd.concat(X_list, axis=0, join="outer").fillna(0.0)
    y_all = pd.concat(y_list, axis=0)
    study_all = pd.concat(study_list, axis=0)

    # align indices
    idx = X_all.index.intersection(y_all.index).intersection(study_all.index)
    X_all = X_all.loc[idx]
    y_all = y_all.loc[idx]
    study_all = study_all.loc[idx]

    X_path = PROCESSED_DIR / "X_all_expanded.csv"
    y_path = PROCESSED_DIR / "y_all_expanded.csv"
    study_path = PROCESSED_DIR / "study_all_expanded.csv"

    X_all.to_csv(X_path)
    y_all.to_frame("label").to_csv(y_path)
    study_all.to_frame("study_id").to_csv(study_path)

    print("\n[INFO] Saved unified processed data:")
    print(f"  Features: {X_path} (shape {X_all.shape})")
    print(f"  Labels:   {y_path} (len {len(y_all)})")
    print(f"  Studies:  {study_path} (unique study_ids: {study_all.nunique()})")


def main() -> None:
    """Entry point so this module can be run as:

        python -m preprocessing.multi_dataset_processor

    This will:
    - Loop over all registered human metabolomics datasets.
    - Build harmonized feature matrices with pathway scores.
    - Save X_all_expanded, y_all_expanded, and study_all_expanded under data/processed/.
    """
    build_multi_dataset_features()


if __name__ == "__main__":
    main()
