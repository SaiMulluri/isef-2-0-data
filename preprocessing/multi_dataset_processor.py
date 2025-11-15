"""Multi-dataset preprocessing pipeline for Alzheimer's metabolomics cohorts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from dataset_registry import REGISTRY


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


DEFAULT_SAMPLE_ID_CANDIDATES = [
    "Sample ID",
    "SampleID",
    "sample_id",
    "Sample",
    "sample",
    "ID",
    "filename",
]

DEFAULT_LABEL_CANDIDATES = [
    "Diagnosis",
    "diagnosis",
    "Group",
    "group",
    "Class",
    "class",
    "Condition",
    "condition",
    "Phenotype",
    "phenotype",
    "Status",
    "status",
]

POS_LABELS = {
    "ad",
    "alz",
    "alzheimers",
    "alzheimer's disease",
    "alzheimers disease",
    "case",
    "patient",
}

NEG_LABELS = {
    "cn",
    "control",
    "cognitively normal",
    "healthy control",
    "hc",
    "normal",
}

EXCLUDED_LABELS = {
    "mci",
    "mild cognitive impairment",
    "other",
    "na",
    "nan",
}

METABOLITE_SYNONYMS: Mapping[str, Iterable[str]] = {
    "tryptophan": {"tryptophan", "l-tryptophan"},
    "indole-3-propionic acid": {
        "indole-3-propionic acid",
        "ipa",
        "indole 3 propionic acid",
    },
    "indole-3-acetic acid": {"indole-3-acetic acid", "indole acetic acid", "iaa"},
    "indole-3-lactic acid": {"indole-3-lactic acid", "indole lactic acid"},
    "kynurenine": {"kynurenine"},
    "kynurenic acid": {"kynurenic acid"},
    "quinolinic acid": {"quinolinic acid"},
    "3-hydroxykynurenine": {"3-hydroxykynurenine", "3 hydroxykynurenine"},
    "tryptamine": {"tryptamine"},
    "acetate": {"acetate", "acetic acid"},
    "propionate": {"propionate", "propionic acid"},
    "butyrate": {"butyrate", "butyric acid"},
    "valerate": {"valerate", "valeric acid"},
    "isobutyrate": {"isobutyrate", "isobutyric acid"},
}

TRP_METABOLITES = {
    "tryptophan",
    "indole-3-propionic acid",
    "indole-3-acetic acid",
    "indole-3-lactic acid",
    "kynurenine",
    "kynurenic acid",
    "quinolinic acid",
    "3-hydroxykynurenine",
    "tryptamine",
}

SCFA_METABOLITES = {
    "acetate",
    "propionate",
    "butyrate",
    "valerate",
    "isobutyrate",
}


@dataclass
class DatasetArtifacts:
    dataset_id: str
    X: pd.DataFrame
    y: pd.Series
    study_ids: pd.Series
    metadata: Dict[str, str]


class MultiDatasetProcessor:
    """Process all registered human metabolomics datasets."""

    def __init__(
        self,
        registry: Optional[Mapping[str, Mapping[str, str]]] = None,
        output_dir: str | Path = "data/processed",
        missing_threshold: float = 0.5,
    ) -> None:
        self.registry = registry or REGISTRY
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.missing_threshold = missing_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_all(self) -> DatasetArtifacts:
        processed: List[DatasetArtifacts] = []
        for dataset_id, config in self.registry.items():
            if config.get("species", "").lower() != "human":
                logger.info("Skipping %s (non-human dataset)", dataset_id)
                continue
            if config.get("data_type", "").lower() != "metabolomics":
                logger.info("Skipping %s (data type not metabolomics)", dataset_id)
                continue

            logger.info("Processing dataset %s", dataset_id)
            artifact = self._process_single_dataset(dataset_id, config)
            if artifact is None:
                logger.warning("Dataset %s could not be processed and will be skipped", dataset_id)
                continue
            processed.append(artifact)

        if not processed:
            raise RuntimeError(
                "No datasets were processed. Ensure raw data is available in the"
                " directories specified in dataset_registry.REGISTRY."
            )

        X_all, y_all, study_all = self._combine_datasets(processed)

        X_path = self.output_dir / "X_all_expanded.csv"
        y_path = self.output_dir / "y_all_expanded.csv"
        study_path = self.output_dir / "study_all_expanded.csv"

        X_all.to_csv(X_path, index=False)
        y_all.to_csv(y_path, index=False, header=["label"])
        study_all.to_csv(study_path, index=False, header=["study_id"])

        logger.info("Saved expanded feature matrix to %s", X_path)
        logger.info("Saved labels to %s", y_path)
        logger.info("Saved study identifiers to %s", study_path)

        return DatasetArtifacts(
            dataset_id="ALL",
            X=X_all,
            y=y_all,
            study_ids=study_all,
            metadata={"processed_datasets": json.dumps([d.dataset_id for d in processed])},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_single_dataset(
        self, dataset_id: str, config: Mapping[str, str]
    ) -> Optional[DatasetArtifacts]:
        raw_dir = Path(config.get("local_raw_dir", ""))
        if not raw_dir.exists():
            logger.warning(
                "Raw directory %s for dataset %s not found. Skipping.", raw_dir, dataset_id
            )
            return None

        data_frame = self._load_raw_data(raw_dir)
        if data_frame is None or data_frame.empty:
            logger.warning("No usable raw data found for dataset %s", dataset_id)
            return None

        sample_col = self._detect_column(data_frame.columns, DEFAULT_SAMPLE_ID_CANDIDATES)
        label_col = self._detect_column(data_frame.columns, DEFAULT_LABEL_CANDIDATES)

        if sample_col is None or label_col is None:
            logger.warning(
                "Dataset %s missing sample or label columns (sample=%s, label=%s).",
                dataset_id,
                sample_col,
                label_col,
            )
            return None

        df = data_frame.copy()
        df = df.dropna(subset=[sample_col])
        df[sample_col] = df[sample_col].astype(str)

        label_series = self._normalize_labels(df[label_col])

        mask_valid = label_series.notna()
        df = df.loc[mask_valid]
        label_series = label_series.loc[mask_valid]

        if label_series.empty:
            logger.warning("Dataset %s has no binary AD/CN labels after filtering", dataset_id)
            return None

        features_df, metadata_df = self._split_features_metadata(df, sample_col, label_col)
        features_df = self._clean_features(features_df)
        features_df = self._harmonize_metabolites(features_df)

        features_df = self._add_pathway_scores(features_df)
        features_df = self._integrate_metadata(features_df, metadata_df)

        features_df[f"study_{dataset_id}"] = 1.0

        features_df.index = df[sample_col]
        label_series.index = features_df.index

        study_ids = pd.Series(dataset_id, index=features_df.index, name="study_id")

        return DatasetArtifacts(
            dataset_id=dataset_id,
            X=features_df,
            y=label_series.astype(int),
            study_ids=study_ids,
            metadata={"n_samples": str(len(features_df))},
        )

    def _load_raw_data(self, raw_dir: Path) -> Optional[pd.DataFrame]:
        candidates = list(raw_dir.glob("**/*"))
        for path in candidates:
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            try:
                if suffix == ".csv":
                    return pd.read_csv(path)
                if suffix in {".tsv", ".txt"}:
                    return pd.read_csv(path, sep="\t")
                if suffix == ".xlsx":
                    return pd.read_excel(path)
                if suffix == ".json":
                    return pd.read_json(path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to read %s: %s", path, exc)
                continue
        return None

    @staticmethod
    def _detect_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
        lower_map = {col.lower(): col for col in columns}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        return None

    @staticmethod
    def _normalize_labels(labels: pd.Series) -> pd.Series:
        def _map_label(value: object) -> Optional[int]:
            if pd.isna(value):
                return None
            if isinstance(value, (int, float)):
                if value in {0, 1}:
                    return int(value)
            normalized = str(value).strip().lower()
            if normalized in POS_LABELS:
                return 1
            if normalized in NEG_LABELS:
                return 0
            if normalized in EXCLUDED_LABELS:
                return None
            return None

        mapped = labels.apply(_map_label)
        return mapped

    def _split_features_metadata(
        self, df: pd.DataFrame, sample_col: str, label_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        metadata_cols = {
            sample_col,
            label_col,
        }
        optional_metadata = [
            "age",
            "Age",
            "AGE",
            "sex",
            "Sex",
            "SEX",
            "gender",
            "Gender",
            "APOE",
            "apoe",
            "apoe_status",
            "mmse",
            "MMSE",
            "cognitive_score",
        ]
        metadata_cols.update({col for col in df.columns if col in optional_metadata})

        metadata_df = df.loc[:, [col for col in df.columns if col in metadata_cols]].copy()
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        features_df = df.loc[:, feature_cols].copy()
        return features_df, metadata_df

    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        numeric_features = features.apply(pd.to_numeric, errors="coerce")
        missing_ratio = numeric_features.isna().mean()
        keep_columns = missing_ratio[missing_ratio <= self.missing_threshold].index.tolist()
        numeric_features = numeric_features[keep_columns]

        if numeric_features.empty:
            return numeric_features

        medians = numeric_features.median()
        numeric_features = numeric_features.fillna(medians)

        numeric_features = np.log1p(numeric_features.clip(lower=0))

        means = numeric_features.mean(axis=0)
        stds = numeric_features.std(axis=0).replace(0, 1)
        numeric_features = (numeric_features - means) / stds
        numeric_features = numeric_features.fillna(0)
        return numeric_features

    def _harmonize_metabolites(self, features: pd.DataFrame) -> pd.DataFrame:
        rename_map: Dict[str, str] = {}
        for column in features.columns:
            lower = column.lower().strip()
            canonical = None
            for canon_name, synonyms in METABOLITE_SYNONYMS.items():
                if lower in synonyms:
                    canonical = canon_name
                    break
            if canonical is not None:
                rename_map[column] = canonical
        harmonized = features.rename(columns=rename_map)
        harmonized = harmonized.groupby(level=0, axis=1).mean()
        return harmonized

    def _add_pathway_scores(self, features: pd.DataFrame) -> pd.DataFrame:
        augmented = features.copy()
        trp_cols = [col for col in augmented.columns if col.lower() in TRP_METABOLITES]
        scfa_cols = [col for col in augmented.columns if col.lower() in SCFA_METABOLITES]

        if trp_cols:
            augmented["TRP_pathway_score"] = augmented[trp_cols].mean(axis=1)
        else:
            augmented["TRP_pathway_score"] = 0.0

        if scfa_cols:
            augmented["SCFA_pathway_score"] = augmented[scfa_cols].mean(axis=1)
        else:
            augmented["SCFA_pathway_score"] = 0.0

        return augmented

    def _integrate_metadata(
        self, features: pd.DataFrame, metadata: pd.DataFrame
    ) -> pd.DataFrame:
        augmented = features.copy()

        if metadata.empty:
            return augmented

        # Age
        age_col = self._detect_column(metadata.columns, ["age"])
        if age_col:
            age_series_raw = pd.to_numeric(metadata[age_col], errors="coerce")
            if age_series_raw.notna().any():
                filled_age = age_series_raw.fillna(age_series_raw.median())
                normalized_age = (filled_age - filled_age.mean()) / filled_age.std(ddof=0)
                normalized_age = normalized_age.fillna(0)
                augmented["age_normalized"] = normalized_age.values
                augmented["age_years"] = filled_age.values

        # Sex / gender
        sex_col = self._detect_column(metadata.columns, ["sex", "gender"])
        if sex_col:
            sex_series = metadata[sex_col].astype(str).str.lower()
            mapped = sex_series.map({"male": "sex_Male", "m": "sex_Male", "female": "sex_Female", "f": "sex_Female"})
            dummies = pd.get_dummies(mapped)
            for col in dummies.columns:
                augmented[col] = dummies[col].astype(float).values

        # APOE genotype
        apoe_col = self._detect_column(metadata.columns, ["apoe", "apoe_status", "apoe_genotype"])
        if apoe_col:
            apoe_series = metadata[apoe_col].astype(str).str.upper()
            dummies = pd.get_dummies(apoe_series, prefix="APOE")
            for col in dummies.columns:
                augmented[col] = dummies[col].astype(float).values

        # Cognitive scores (e.g., MMSE)
        mmse_col = self._detect_column(metadata.columns, ["mmse", "cognitive_score"])
        if mmse_col:
            mmse_series = pd.to_numeric(metadata[mmse_col], errors="coerce")
            if mmse_series.notna().any():
                mmse_series = mmse_series.fillna(mmse_series.median())
                mmse_series = (mmse_series - mmse_series.mean()) / mmse_series.std(ddof=0)
                mmse_series = mmse_series.fillna(0)
                augmented[f"{mmse_col}_normalized"] = mmse_series.values

        return augmented

    def _combine_datasets(
        self, datasets: List[DatasetArtifacts]
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        all_columns = sorted({col for ds in datasets for col in ds.X.columns})
        combined_features = []
        combined_labels = []
        combined_studies = []

        for ds in datasets:
            df = ds.X.reindex(columns=all_columns, fill_value=0)
            combined_features.append(df)
            combined_labels.append(ds.y)
            combined_studies.append(ds.study_ids)

        X_all = pd.concat(combined_features, axis=0)
        y_all = pd.concat(combined_labels, axis=0)
        study_all = pd.concat(combined_studies, axis=0)

        return X_all.reset_index(drop=True), y_all.reset_index(drop=True), study_all.reset_index(
            drop=True
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    processor = MultiDatasetProcessor()
    try:
        processor.process_all()
    except RuntimeError as exc:
        logger.error("Processing failed: %s", exc)


if __name__ == "__main__":
    main()

