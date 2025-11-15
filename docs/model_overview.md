# High-capacity Alzheimer’s metabolomics modeling framework

This document summarizes the architecture of the registry-driven pipeline that
scales to many Alzheimer’s-related metabolomics cohorts.

## Dataset registry

- [`dataset_registry.py`](../dataset_registry.py) defines a single `REGISTRY`
  mapping of dataset identifiers to metadata (species, matrix, local directories,
  notes, etc.).
- All ingestion, preprocessing, modeling, and evaluation modules query the
  registry to discover available datasets—no scripts hard-code study IDs.
- Adding new cohorts (human or mouse) only requires inserting a new entry in the
  registry and placing the raw data in the specified `local_raw_dir` path.

## Preprocessing pipeline

- [`preprocessing/multi_dataset_processor.py`](../preprocessing/multi_dataset_processor.py)
  orchestrates ingestion of **all registered human metabolomics datasets**.
- Key responsibilities:
  - Load raw files from each dataset directory (CSV/TSV/XLSX/JSON supported).
  - Detect sample IDs and diagnosis columns, normalize binary AD vs CN labels,
    and drop other classes (e.g., MCI) from training labels.
  - Convert metabolite intensities to numeric values, remove high-missingness
    features, impute with per-feature medians, apply log1p and z-score
    transformations.
  - Harmonize metabolite names using pathway-aware synonym tables for the
    tryptophan (TRP) and short-chain fatty acid (SCFA) pathways.
  - Engineer pathway scores (TRP and SCFA composite metrics) and integrate
    optional clinical covariates (age, sex, APOE, cognitive scores). Covariates
    are automatically standardized/encoded when present and ignored otherwise.
  - Append one-hot study indicators (e.g., `study_ST000046`) to support domain
    adaptation techniques downstream.
  - Concatenate all cohort-specific matrices into `X_all_expanded.csv`,
    alongside aligned labels (`y_all_expanded.csv`) and study identifiers
    (`study_all_expanded.csv`). Missing features for a given cohort are filled
    with zeros so that the combined matrix has consistent columns across all
    studies.

## Model training and evaluation

- [`train_high_capacity_models.py`](../train_high_capacity_models.py) performs
  leave-one-study-out (LOSO) cross-validation across every unique study ID.
  - During each LOSO split it trains multiple model families (Random Forest,
    HistGradientBoosting, Logistic Regression, and optional XGBoost/LightGBM when
    installed).
  - Random Forest receives lightweight randomized hyperparameter tuning.
  - Evaluation metrics (accuracy, ROC AUC, precision, recall, F1, confusion
    counts) are recorded per model per held-out cohort in
    `high_capacity_eval_by_study.csv`.
  - Summary statistics (mean/std of accuracy and ROC AUC) are written to
    `high_capacity_eval_summary.json`.
  - The model type with the best mean ROC AUC (falling back to accuracy if
    necessary) is retrained on the full dataset and saved to
    `models/high_capacity_metabolomics_model.pkl`.

## Subgroup and cohort assessment

- [`analyze_subgroups_and_cohorts.py`](../analyze_subgroups_and_cohorts.py)
  evaluates the final model across:
  - Each registered cohort (cohort-level metrics).
  - Demographic subgroups (e.g., age < 70 vs ≥ 70, sex-specific groups) when the
    requisite metadata exists in the feature matrix.
- Metrics include accuracy, ROC AUC, sensitivity, and specificity, with results
  saved to `data/processed/subgroup_and_cohort_performance.csv`.

## Scaling guidance

- The entire pipeline is designed to handle 10–30 (or more) cohorts without code
  changes. Extending the analysis to additional studies requires only registry
  updates and raw data placement.
- Mouse datasets can be included in the registry for downstream analyses or
  stratified evaluations, while the core training pipeline currently focuses on
  human metabolomics cohorts.

## Safety disclaimer

This repository is intended solely for exploratory research. The models and
metrics generated here **must not** be used for clinical decision making,
diagnosis, or patient management. Any deployment in healthcare settings would
require rigorous validation, regulatory approval, and supervision by qualified
medical professionals.
