"""Utilities for injecting labels into Metabolomics Workbench datatables.

This module focuses on MW studies where the exported datatables only contain
metabolite intensities and sample identifiers. For those cohorts, the matching
``*_mwtab.txt`` file usually embeds sample-level metadata (e.g., Genotype or
Diagnosis) that can be repurposed as case/control labels. The helper below
parses that metadata, aligns it with the datatable by ``Sample_ID``, and
returns a dataframe that now includes a label-bearing column.

Usage
-----
The helpers are designed to be called automatically from
``label_and_merge_datasets.py`` but can also be used directly:

    python - <<'PY'
    from pathlib import Path
    from preprocessing.mwtab_labeler import MW_DATASET_CONFIG, attach_labels_to_datatable

    study_id = "ST000462"
    datatable = Path(f"data/metabolomics_workbench/{study_id}/{study_id}_datatable.txt")
    df, label_col = attach_labels_to_datatable(study_id, datatable, MW_DATASET_CONFIG[study_id])
    print(label_col, df[label_col].value_counts(dropna=False))
    PY
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


DEFAULT_POSITIVE_KEYWORDS = ["alzheim", "ad", "case", "app", "tg", "mci"]
DEFAULT_NEGATIVE_KEYWORDS = ["control", "cn", "healthy", "normal", "ntg", "wt"]


@dataclass
class MwStudyConfig:
    """Configuration for a single MW study.

    Attributes
    ----------
    sample_column:
        Column name in the datatable representing sample identifiers.
    label_candidates:
        Ordered preference list of label-bearing columns to look for.
    positive_keywords / negative_keywords:
        Lists of lowercase substrings that map to binary labels.
    metadata_source:
        If set to "mwtab", attempt to recover labels from the corresponding
        ``*_mwtab.txt`` file when the datatable is missing a usable label.
    metadata_label_field:
        Field name within the parsed metadata to use as the label column.
    metadata_sample_field:
        Field name within the parsed metadata that corresponds to
        ``sample_column`` in the datatable.
    """

    sample_column: str
    label_candidates: List[str]
    positive_keywords: List[str]
    negative_keywords: List[str]
    metadata_source: Optional[str] = None
    metadata_label_field: Optional[str] = None
    metadata_sample_field: str = "Sample_ID"


MW_DATASET_CONFIG: Dict[str, MwStudyConfig] = {
    "ST000046": MwStudyConfig(
        sample_column="Sample_ID",
        label_candidates=["Group", "Condition", "Diagnosis", "Class"],
        positive_keywords=["alzheim", "ad", "mci"],
        negative_keywords=["control", "cn", "healthy", "normal"],
    ),
    "ST000047": MwStudyConfig(
        sample_column="Sample_ID",
        label_candidates=["Diagnosis", "Group", "Condition", "Class"],
        positive_keywords=["alzheim", "ad", "mci"],
        negative_keywords=["control", "cn", "healthy", "normal"],
    ),
    "ST000462": MwStudyConfig(
        sample_column="Sample_ID",
        label_candidates=["Genotype", "Group", "Condition", "Class"],
        positive_keywords=["app"],
        negative_keywords=["ntg", "control", "wt"],
        metadata_source="mwtab",
        metadata_label_field="Genotype",
    ),
    "ST001152": MwStudyConfig(
        sample_column="Sample_ID",
        label_candidates=["Group", "Condition", "Diagnosis", "Class"],
        positive_keywords=DEFAULT_POSITIVE_KEYWORDS,
        negative_keywords=DEFAULT_NEGATIVE_KEYWORDS,
        metadata_source="mwtab",
        metadata_label_field="Condition",
    ),
    "ST001050": MwStudyConfig(
        sample_column="Sample_ID",
        label_candidates=["Group", "Condition", "Diagnosis", "Class"],
        positive_keywords=DEFAULT_POSITIVE_KEYWORDS,
        negative_keywords=DEFAULT_NEGATIVE_KEYWORDS,
        metadata_source="mwtab",
        metadata_label_field="Condition",
    ),
}


def _extract_subject_sample_metadata(mwtab_path: Path) -> pd.DataFrame:
    """Parse a mwTab text file for SUBJECT_SAMPLE_FACTORS.

    The mwTab format is pseudo-JSON; we use a lightweight regex to capture each
    block that looks like::

        {"Subject ID":"-","Sample ID":"BP_1","Factors":{"Age":"Pool", ...}}

    Parameters
    ----------
    mwtab_path:
        Path to the mwTab file.

    Returns
    -------
    DataFrame
        Columns include ``Sample_ID`` and any factor keys discovered (e.g.,
        Genotype, Age, Gender, Diet).
    """

    import re

    text = mwtab_path.read_text(errors="ignore")
    pattern = re.compile(
        r"\{\s*\"Subject ID\".*?\"Sample ID\"\s*:\s*\"(?P<sample>[^\"]+)\""
        r".*?\"Factors\"\s*:\s*\{(?P<factors>[^\}]*)\}"
        r"\s*(?:,\s*\"Additional sample data\"\s*:\s*\{(?P<extra>[^\}]*)\})?\s*\}",
        re.DOTALL,
    )

    records: List[Dict[str, str]] = []
    for match in pattern.finditer(text):
        factors_raw = match.group("factors") or ""
        factor_parts = [p for p in factors_raw.split(",") if ":" in p]
        factor_dict: Dict[str, str] = {}
        for part in factor_parts:
            key, value = part.split(":", 1)
            factor_dict[key.strip().strip('"')] = value.strip().strip('"')
        record = {"Sample_ID": match.group("sample"), **factor_dict}
        records.append(record)

    return pd.DataFrame(records)


def _locate_label_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
        for col in df.columns:
            if cand.lower() == col.lower() or cand.lower() in col.lower():
                return col
    return None


def _map_labels(values: pd.Series, positive_keywords: List[str], negative_keywords: List[str]) -> pd.Series:
    mapped: List[Optional[int]] = []
    pos_lower = [k.lower() for k in positive_keywords]
    neg_lower = [k.lower() for k in negative_keywords]

    for raw in values.fillna(""):
        val = str(raw).lower()
        label: Optional[int] = None
        if any(k in val for k in pos_lower):
            label = 1
        elif any(k in val for k in neg_lower):
            label = 0
        mapped.append(label)

    return pd.Series(mapped, index=values.index)


def attach_labels_to_datatable(
    study_id: str, datatable_path: Path, config: MwStudyConfig
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Ensure the datatable for ``study_id`` contains a label-bearing column.

    Returns the dataframe (unmodified or augmented) along with the name of the
    column that should be used for label mapping.
    """

    # Try a conservative read first (ignoring comment lines). Some datatables
    # are shipped with leading "#" metadata rows which, when combined with
    # ``comment="#"``, can yield an empty frame. If that happens, retry without
    # treating "#" as a comment so we still parse the tabular content.
    df = pd.read_csv(datatable_path, sep="\t", dtype=str, comment="#")
    if df.empty:
        df = pd.read_csv(datatable_path, sep="\t", dtype=str, comment=None)

    label_col = _locate_label_column(df, config.label_candidates)
    if label_col:
        return df, label_col

    if config.metadata_source == "mwtab":
        mwtab_path = datatable_path.with_name(f"{study_id}_mwtab.txt")
        if not mwtab_path.exists():
            print(f"Metadata mwTab missing for {study_id}: {mwtab_path}")
            return df, label_col

        metadata_df = _extract_subject_sample_metadata(mwtab_path)
        if metadata_df.empty:
            print(f"No metadata parsed from {mwtab_path} for {study_id}")
            return df, label_col

        label_field = config.metadata_label_field or "Condition"
        sample_field = config.metadata_sample_field
        if label_field not in metadata_df.columns:
            print(
                f"Metadata label field '{label_field}' not found for {study_id};"
                f" available columns: {list(metadata_df.columns)}"
            )
            return df, label_col

        merged = df.merge(
            metadata_df[[sample_field, label_field]],
            left_on=config.sample_column,
            right_on=sample_field,
            how="left",
        )

        label_col = label_field
        if label_col not in merged.columns:
            return df, None
        return merged, label_col

    # No metadata source and no in-datatable label column.
    return df, label_col


__all__ = [
    "MwStudyConfig",
    "MW_DATASET_CONFIG",
    "attach_labels_to_datatable",
]

