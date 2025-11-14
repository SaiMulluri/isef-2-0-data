# ISEF 2.0 Data: Alzheimer–Microbiome Raw Datasets

This repository provides scripts and tooling to download publicly available
Alzheimer-related metabolomics and microbiome datasets, preprocess and label the
metabolomics data, and train machine learning models for early detection.

## Usage

1. **Install dependencies**

   ```bash
   python3 -m venv .venv  # optional
   source .venv/bin/activate  # optional
   pip3 install -r requirements.txt
   ```

2. **Run the data acquisition and ML pipeline**

   ```bash
   python3 download_alz_datasets.py
   python3 label_and_merge_datasets.py
   python3 train_alz_microbiome_model.py
   ```

3. **Outputs**
   - Processed metabolomics feature matrix and labels are saved in
     `data/processed/` as `X_metabolomics.csv` and `y_metabolomics.csv`.
   - Trained models, scaler, and report are saved in the `models/` directory.
