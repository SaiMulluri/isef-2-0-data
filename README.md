# ISEF 2.0 Data: Alzheimer–Microbiome Raw Datasets

This repository provides scripts and tooling to download publicly available
Alzheimer-related metabolomics and microbiome datasets, preprocess and label the
metabolomics data, and train machine learning models for early detection.

The project has been extended into a scalable, registry-driven research
framework that can ingest many Alzheimer's cohorts (human and mouse), build
harmonized feature sets, and run leave-one-study-out model evaluations without
hard-coding any study identifiers.

> **Critical disclaimer:** This repository is a research framework only. It is
> not a clinical diagnostic tool and must not be used for screening, diagnosis,
> treatment decisions, or any medical guidance. Even comprehensive benchmarking
> across many cohorts cannot replace prospective clinical validation, regulatory
> review, and physician oversight.

## Usage

1. **Install dependencies**

   ```bash
   python3 -m venv .venv  # optional
   source .venv/bin/activate  # optional
   pip3 install -r requirements.txt
   ```

2. **Register datasets**

   - Update [`dataset_registry.py`](dataset_registry.py) with the metadata for
     any cohorts you want to include. Each entry specifies the dataset ID,
     species, matrix, and local data directories. The preprocessing and modeling
     code automatically iterates over every human metabolomics dataset in this
     registry, so expanding the study list simply requires adding entries—no
     other code changes are necessary.

3. **Confirm raw data availability**

   ```bash
   python download_registered_datasets.py
   ```

   The script will report which datasets have files available locally and which
   still require manual downloads (e.g., from Metabolomics Workbench). No
   automated downloading from external URLs is performed.

4. **Run the high-capacity multi-cohort pipeline**

   ```bash
   python -m preprocessing.multi_dataset_processor
   python train_high_capacity_models.py
   python analyze_subgroups_and_cohorts.py
   ```

   These steps will:

   - Ingest every registered human metabolomics dataset, harmonize metabolite
     names (including pathway-specific synonyms), engineer pathway scores
     (tryptophan, SCFAs, etc.), integrate optional clinical covariates, and
     combine them into an expanded feature matrix (`data/processed/X_all_expanded.csv`).
   - Train multiple machine learning models under leave-one-study-out cross
     validation across all available cohorts and summarize the metrics in
     `data/processed/high_capacity_eval_by_study.csv` and
     `data/processed/high_capacity_eval_summary.json`.
   - Retrain the best-performing model on the full dataset and evaluate it on
     demographic subgroups (age, sex) and each cohort individually, saving the
     results to `data/processed/subgroup_and_cohort_performance.csv` and the
     model artifact to `models/high_capacity_metabolomics_model.pkl`.

5. **Legacy scripts**

   Original scripts for single-cohort analyses remain available (e.g.,
   `process_metabolomics.py`, `train_alz_metabolomics_model.py`). They can be
   used for reproducing earlier baseline experiments but are superseded by the
   multi-cohort workflow above.

## Scaling to many cohorts

- The dataset registry is designed to grow to 10–30 (or more) Alzheimer's
  metabolomics cohorts, including optional mouse datasets.
- Adding a new cohort requires only:
  1. Inserting a new entry in `dataset_registry.REGISTRY` with the appropriate
     metadata and local data paths.
  2. Placing the raw files under the specified `local_raw_dir`.
  3. Re-running the three pipeline commands listed above.
- The preprocessing, modeling, and evaluation code automatically expands to the
  new cohorts—no study IDs are hard-coded anywhere in the pipeline.

## Additional documentation

- [`docs/model_overview.md`](docs/model_overview.md) describes the architecture,
  preprocessing logic, modeling strategy, and evaluation plan in greater
  detail.
