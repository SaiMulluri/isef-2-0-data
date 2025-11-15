"""Central dataset registry for Alzheimer's-related metabolomics studies.

The registry is the single source of truth for dataset metadata that powers
dataset discovery, preprocessing, modeling, and evaluation across the
framework. New cohorts can be onboarded by adding a new entry to
``REGISTRY``; all downstream scripts dynamically iterate over this mapping
without any hard-coded study identifiers.
"""

from __future__ import annotations

from typing import Dict, Mapping, MutableMapping


DatasetEntry = Dict[str, str]


# NOTE: Additional cohorts can be appended to this mapping without modifying
# any other part of the codebase. Downstream components automatically adapt to
# the available datasets.
REGISTRY: MutableMapping[str, DatasetEntry] = {
    "ST000046": {
        "dataset_id": "ST000046",
        "data_type": "metabolomics",
        "species": "human",
        "matrix": "plasma",
        "local_raw_dir": "data/raw/ST000046",
        "local_processed_prefix": "data/processed/ST000046",
        "notes": "Human plasma metabolomics, AD vs MCI vs CN",
    },
    "ST000047": {
        "dataset_id": "ST000047",
        "data_type": "metabolomics",
        "species": "human",
        "matrix": "plasma",
        "local_raw_dir": "data/raw/ST000047",
        "local_processed_prefix": "data/processed/ST000047",
        "notes": "Human plasma metabolomics, AD vs CN",
    },
    "ST000462": {
        "dataset_id": "ST000462",
        "data_type": "metabolomics",
        "species": "human",
        "matrix": "serum",
        "local_raw_dir": "data/raw/ST000462",
        "local_processed_prefix": "data/processed/ST000462",
        "notes": "Serum metabolomics with Alzheimer\'s and control subjects",
    },
    "ST001152": {
        "dataset_id": "ST001152",
        "data_type": "metabolomics",
        "species": "human",
        "matrix": "plasma",
        "local_raw_dir": "data/raw/ST001152",
        "local_processed_prefix": "data/processed/ST001152",
        "notes": "Targeted metabolomics with cognitive assessments",
    },
    "ST001050": {
        "dataset_id": "ST001050",
        "data_type": "metabolomics",
        "species": "human",
        "matrix": "serum",
        "local_raw_dir": "data/raw/ST001050",
        "local_processed_prefix": "data/processed/ST001050",
        "notes": "Serum metabolomics profiling with AD phenotypes",
    },
    # Placeholder entries for additional studies can be added later, including
    # mouse metabolomics cohorts or other modalities (proteomics, microbiome,
    # etc.).
    # Example:
    # "ST001848": {
    #     "dataset_id": "ST001848",
    #     "data_type": "metabolomics",
    #     "species": "mouse",
    #     "matrix": "brain",
    #     "local_raw_dir": "data/raw/ST001848",
    #     "local_processed_prefix": "data/processed/ST001848",
    #     "notes": "Mouse model metabolomics for Alzheimer\'s pathology",
    # },
}


def get_registry() -> Mapping[str, DatasetEntry]:
    """Return an immutable view of the dataset registry."""

    return dict(REGISTRY)


__all__ = ["DatasetEntry", "REGISTRY", "get_registry"]

