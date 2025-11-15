"""Process and harmonise multiple Metabolomics Workbench Alzheimer cohorts.

Datasets
========
The pipeline uses five human metabolomics studies that profile Alzheimer's
disease (AD) patients and cognitively normal (CN) controls:

* ST000046 — Mayo Clinic plasma cohort
* ST000047 — Mayo Clinic cerebrospinal fluid cohort
* ST000462 — Alzheimer's plasma metabolomics
* ST001152 — Serum cohort with AD/CN annotations
* ST001050 — Plasma metabolomics focusing on neurodegeneration

The goal is to build a consistent sample × metabolite matrix across the cohorts
and to engineer pathway-level summary scores for the tryptophan (TRP) and
short-chain fatty acid (SCFA) pathways.  The resulting harmonised data will feed
into a leave-one-study-out modelling workflow implemented in
``train_multi_cohort_metabolomics_model.py``.

Label strategy
--------------
Cognitive status is pulled from each study's mwtab metadata ``SUBJECT_SAMPLE``
section.  We map string values to binary labels as follows:

* ``{"AD", "ALZHEIMER", "ALZHEIMER'S", "PROBABLE AD"}`` → 1
* ``{"CN", "CONTROL", "COGNITIVELY NORMAL", "HEALTHY"}`` → 0

Samples annotated as ``MCI`` (mild cognitive impairment), ``POSSIBLE AD`` or any
other ambiguous/unknown status are omitted.  Inline comments in the helper
functions document the mapping for each study.

Processing steps
----------------
1. Load each measurement table (mwtab ``datatable`` exports or ``*_file_*.txt``
   files).
2. Convert metabolite intensities to floats, drop columns with >40% missing
   values, impute remaining NaNs with the median, apply ``log1p`` and then
   z-score within each study.
3. Harmonise metabolite names across cohorts using a hand-curated synonym map for
   TRP-pathway analytes.
4. Restrict to metabolites observed in every cohort and add pathway summary
   scores for TRP and SCFA metabolites (per-sample means of the available
   z-scored metabolites).
5. Concatenate cohorts together, append one-hot study indicators, and write both
   per-study and combined matrices to ``data/processed``.

Run from the repository root::

   python process_multi_metabolomics.py
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
STUDY_IDS: Tuple[str, ...] = ("ST000046", "ST000047", "ST000462", "ST001152", "ST001050")
MAX_MISSING_FRACTION = 0.5

TRP_SYNONYM_MAP: Dict[str, str] = {
    "indole-3-propionic acid": "Indole-3-propionic acid",
    "indolepropionic acid": "Indole-3-propionic acid",
    "indole-3-propionate": "Indole-3-propionic acid",
    "tryptophan": "Tryptophan",
    "l-tryptophan": "Tryptophan",
    "kynurenine": "Kynurenine",
    "l-kynurenine": "Kynurenine",
    "kynurenic acid": "Kynurenic acid",
    "quinolinic acid": "Quinolinic acid",
    "3-hydroxykynurenine": "3-Hydroxykynurenine",
    "3 hydroxykynurenine": "3-Hydroxykynurenine",
    "indole-3-acetic acid": "Indole-3-acetic acid",
    "indoleacetic acid": "Indole-3-acetic acid",
    "indole-3-lactic acid": "Indole-3-lactic acid",
    "tryptamine": "Tryptamine",
}

TRYPTOPHAN_PATHWAY: List[str] = [
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

LABEL_POSITIVE = {"AD", "ALZHEIMER", "ALZHEIMER'S", "PROBABLE AD"}
LABEL_NEGATIVE = {"CN", "CONTROL", "COGNITIVELY NORMAL", "HEALTHY"}
LABEL_EXCLUDE = {"MCI", "POSSIBLE AD", "UNKNOWN", "N/A"}
FACTOR_KEYS = [
    "Cognitive Status",
    "Diagnosis",
    "Disease",
    "Condition",
    "Group",
    "Phenotype",
    "Clinical status",
]


@dataclass
class StudyData:
    """Container for the processed outputs of a single study."""

    X: pd.DataFrame
    y: pd.Series
    study_ids: pd.Series


def strip_html_shell(text: str) -> str:
    """Remove the minimal ``<html><body><pre>`` shell sometimes added to tables."""

    return text.replace("<html><body><pre>", "").replace("</pre></body></html>", "")


def detect_delimiter(text: str) -> str:
    """Heuristically detect the delimiter for a Workbench export."""

    tab_count = text.count("\t")
    comma_count = text.count(",")
    return "\t" if tab_count >= comma_count else ","


def find_measurement_files(study_id: str) -> List[Path]:
    """Return candidate measurement files for ``study_id`` in priority order."""

    study_dir = RAW_DIR / study_id
    if not study_dir.exists():
        raise FileNotFoundError(f"Study directory {study_dir} does not exist. Run download script first.")

    candidates = []
    datatable = study_dir / f"{study_id}_datatable.txt"
    if datatable.exists():
        candidates.append(datatable)

    mwtab_table = study_dir / f"{study_id}_mwtab.txt"
    if mwtab_table.exists():
        candidates.append(mwtab_table)

    candidates.extend(sorted(study_dir.glob(f"{study_id}_file_*.txt")))

    other_txt = [p for p in study_dir.glob("*.txt") if p not in candidates]
    candidates.extend(sorted(other_txt))
    return candidates


def parse_measurement_file(path: Path) -> pd.DataFrame:
    """Parse a Workbench tabular export into a sample × metabolite frame."""

    text = strip_html_shell(path.read_text(encoding="utf-8"))
    delimiter = detect_delimiter(text)
    table = pd.read_csv(StringIO(text), sep=delimiter, comment="#", dtype=str)
    table = table.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)

    if {"Metabolite_name", "RefMet_name"}.issubset(table.columns):
        # Standard untargeted export (metabolites as rows, samples as columns)
        metabolite_names = table["Metabolite_name"].fillna("").astype(str).str.strip()
        refmet_names = table["RefMet_name"].fillna("").astype(str).str.strip()
        canonical = np.where(refmet_names.eq(""), metabolite_names, refmet_names)
        canonical = pd.Series(canonical, name="metabolite")
        data = table.drop(columns=["Metabolite_name", "RefMet_name"], errors="ignore")
        data.index = canonical
        data = data.apply(pd.to_numeric, errors="coerce")
        return data.transpose()

    # Some datatables already come with samples as rows and metabolite columns.
    sample_candidates = [c for c in table.columns if "sample" in c.lower()]
    if sample_candidates:
        table = table.set_index(sample_candidates[0])
    table.index = table.index.astype(str).str.strip()
    table = table.apply(pd.to_numeric, errors="coerce")
    return table


def load_labels_from_mwtab(study_id: str) -> Dict[str, str]:
    """Extract sample-level diagnoses from the mwtab metadata file."""

    mwtab_path = RAW_DIR / study_id / f"{study_id}_mwtab.txt"
    if not mwtab_path.exists():
        logging.warning("mwtab metadata missing for %s; unable to derive labels", study_id)
        return {}

    text = mwtab_path.read_text(encoding="utf-8")
    samples: Dict[str, str] = {}
    for match in re.finditer(r'"Sample ID"\s*:\s*"(?P<sample>[^"]+)"\s*,\s*"Factors"\s*:\s*\{(?P<factors>[^}]*)\}', text, re.IGNORECASE):
        sample_id = match.group("sample").strip()
        factors_text = match.group("factors")
        status = None
        for key in FACTOR_KEYS:
            pattern = re.compile(rf'"{re.escape(key)}"\s*:\s*"(?P<value>[^"]+)"', re.IGNORECASE)
            factor_match = pattern.search(factors_text)
            if factor_match:
                status = factor_match.group("value").strip()
                break
        if status is None:
            continue
        samples[sample_id] = status
    return samples


def map_status_to_label(status: str) -> Optional[int]:
    """Normalise a textual status to binary label or return ``None`` to drop."""

    status_clean = status.strip().upper()
    if status_clean in LABEL_POSITIVE:
        return 1
    if status_clean in LABEL_NEGATIVE:
        return 0
    if status_clean in LABEL_EXCLUDE:
        return None
    # Try to standardise simple variations (e.g., Control, Alzheimer disease).
    tokens = re.sub(r"[^A-Z]", " ", status_clean).split()
    if any(token in {"CONTROL", "CN", "HEALTHY"} for token in tokens):
        return 0
    if any(token in {"AD", "ALZHEIMER", "ALZHEIMERS"} for token in tokens):
        return 1
    return None


def canonicalise_columns(columns: Iterable[str]) -> List[str]:
    """Apply the TRP synonym map to metabolite names."""

    canonical: List[str] = []
    for col in columns:
        key = col.strip()
        canonical_name = TRP_SYNONYM_MAP.get(key.lower(), key)
        canonical.append(canonical_name)
    return canonical


def clean_study_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Apply numeric conversion, missingness filtering, log1p, and z-scoring."""

    X_numeric = X.apply(pd.to_numeric, errors="coerce")
    missing_fraction = X_numeric.isna().mean(axis=0)
    keep_cols = missing_fraction <= MAX_MISSING_FRACTION
    X_filtered = X_numeric.loc[:, keep_cols]

    if X_filtered.empty:
        raise ValueError("All metabolites were dropped due to missingness; adjust threshold or inspect data.")

    imputer = SimpleImputer(strategy="median")
    imputed = pd.DataFrame(imputer.fit_transform(X_filtered), index=X_filtered.index, columns=X_filtered.columns)
    imputed = imputed.clip(lower=0)
    log_transformed = np.log1p(imputed)
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(log_transformed), index=log_transformed.index, columns=log_transformed.columns)
    return scaled


def compute_pathway_score(X: pd.DataFrame, metabolites: Sequence[str]) -> pd.Series:
    """Compute the mean z-score across available metabolites in ``metabolites``."""

    available = [m for m in metabolites if m in X.columns]
    if not available:
        return pd.Series(data=np.nan, index=X.index, name="pathway_score")
    return X[available].mean(axis=1)


def process_study(study_id: str) -> StudyData:
    """Load, clean, and label a single study."""

    logging.info("Processing %s", study_id)
    measurement_frames: List[pd.DataFrame] = []
    for path in find_measurement_files(study_id):
        try:
            frame = parse_measurement_file(path)
        except Exception as exc:  # noqa: BLE001 - propagate informative context
            logging.debug("Skipping %s due to parse error: %s", path, exc)
            continue
        measurement_frames.append(frame)

    if not measurement_frames:
        raise RuntimeError(f"No parsable measurement files found for {study_id}.")

    X = pd.concat(measurement_frames, axis=1)
    X.columns = canonicalise_columns(X.columns)
    X = X[~X.index.duplicated(keep="first")]

    raw_labels = load_labels_from_mwtab(study_id)
    if not raw_labels:
        raise RuntimeError(f"No labels extracted for {study_id}; ensure mwtab metadata exists.")

    y_records = {}
    for sample_id, status in raw_labels.items():
        label = map_status_to_label(status)
        if label is None:
            continue
        y_records[sample_id.strip()] = label

    overlapping_samples = sorted(set(X.index) & set(y_records.keys()))
    if not overlapping_samples:
        raise RuntimeError(f"No overlapping samples between measurements and labels for {study_id}.")

    X = X.loc[overlapping_samples]
    y = pd.Series({sample: y_records[sample] for sample in overlapping_samples}, name="label")
    study_series = pd.Series(study_id, index=X.index, name="study_id")

    X_clean = clean_study_matrix(X)
    return StudyData(X=X_clean, y=y, study_ids=study_series)


def harmonise_and_save(studies: List[StudyData]) -> None:
    """Intersect features, add pathway scores, append study dummies, and save files."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    feature_sets = [set(study.X.columns) for study in studies]
    common_features = set.intersection(*feature_sets)
    if not common_features:
        raise RuntimeError("No common metabolites shared across all studies.")
    common_features = sorted(common_features)

    X_all_frames: List[pd.DataFrame] = []
    y_all = []
    study_all = []

    for study in studies:
        X_subset = study.X.loc[:, common_features].copy()
        trp_score = compute_pathway_score(X_subset, TRYPTOPHAN_PATHWAY)
        scfa_score = compute_pathway_score(X_subset, SCFA_METABS)
        X_subset["TRP_pathway_score"] = trp_score
        X_subset["SCFA_pathway_score"] = scfa_score

        study_dummy = pd.get_dummies(study.study_ids, prefix="study")
        X_augmented = pd.concat([X_subset, study_dummy], axis=1)

        X_all_frames.append(X_augmented)
        y_all.append(study.y)
        study_all.append(study.study_ids)

        # Save per-study matrices for inspection
        X_path = PROCESSED_DIR / f"X_{study.study_ids.iloc[0]}.csv"
        y_path = PROCESSED_DIR / f"y_{study.study_ids.iloc[0]}.csv"
        X_augmented.to_csv(X_path, index_label="sample_id")
        study.y.to_csv(y_path, header=True)

    X_all = pd.concat(X_all_frames, axis=0).fillna(0.0)
    y_all_series = pd.concat(y_all, axis=0)
    study_all_series = pd.concat(study_all, axis=0)

    X_all.to_csv(PROCESSED_DIR / "X_all_metabolomics.csv", index_label="sample_id")
    y_all_series.to_csv(PROCESSED_DIR / "y_all_metabolomics.csv", header=True)
    study_all_series.to_csv(PROCESSED_DIR / "study_all_metabolomics.csv", header=True)


def main() -> None:
    """Process all configured studies and write harmonised matrices."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    studies: List[StudyData] = []
    for study_id in STUDY_IDS:
        try:
            study_data = process_study(study_id)
        except Exception as exc:  # noqa: BLE001 - include study context
            logging.error("Failed to process %s: %s", study_id, exc)
            raise
        studies.append(study_data)

    harmonise_and_save(studies)
    logging.info("Saved combined metabolomics matrices to %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()
