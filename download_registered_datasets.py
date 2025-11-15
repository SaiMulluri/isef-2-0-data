"""Utility script to validate availability of registered datasets.

The script inspects ``dataset_registry.REGISTRY`` and reports whether the
expected raw data directory for each cohort already contains files. It does not
perform any automatic downloading; researchers should obtain the data directly
from trusted repositories such as Metabolomics Workbench.
"""

from __future__ import annotations

from pathlib import Path

from dataset_registry import REGISTRY


def check_dataset_availability(dataset_id: str, config: dict) -> None:
    raw_dir = Path(config["local_raw_dir"]) if config.get("local_raw_dir") else None
    if raw_dir is None:
        print(f"[WARN] Dataset {dataset_id} does not define a local_raw_dir.")
        return

    if not raw_dir.exists():
        print(
            f"[TODO] Raw directory {raw_dir} for {dataset_id} does not exist."
        )
        print(
            "       Please download the dataset manually (e.g., from Metabolomics"
            " Workbench) and place it in the directory."
        )
        return

    data_files = [
        p
        for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".txt", ".xlsx", ".json"}
    ]

    if data_files:
        print(
            f"[OK] Found raw data for {dataset_id} in {raw_dir}. "
            f"Detected files: {[p.name for p in data_files][:5]}"
        )
    else:
        print(
            f"[TODO] No recognized data files found in {raw_dir} for {dataset_id}."
        )
        print(
            "       Ensure that raw files (CSV/TSV/TXT/XLSX/JSON) are placed in the"
            " directory."
        )


def main() -> None:
    print("Checking availability of registered datasets...")
    for dataset_id, config in REGISTRY.items():
        check_dataset_availability(dataset_id, config)


if __name__ == "__main__":
    main()

