"""Download Alzheimer-related metabolomics and microbiome datasets.

This script fetches study tables from the Metabolomics Workbench REST API and
retrieves run metadata from the ENA portal API. Files are saved into the
repository's data directory structure for downstream processing.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Sequence, Set

import requests


MW_BASE_URL = "https://www.metabolomicsworkbench.org/rest"
ENA_BASE_URL = "https://www.ebi.ac.uk/ena/portal/api/search"
FORCED_STUDY_IDS = {"ST000046", "ST000047", "ST000462"}
DATA_DIR = Path("data")
MW_DIR = DATA_DIR / "metabolomics_workbench"
ENA_DIR = DATA_DIR / "ena"


def ensure_directories() -> None:
    """Ensure required output directories exist."""
    MW_DIR.mkdir(parents=True, exist_ok=True)
    ENA_DIR.mkdir(parents=True, exist_ok=True)


def _extract_study_ids_from_json(obj: object) -> Set[str]:
    """Attempt to extract study IDs from a JSON-like response."""
    study_ids: Set[str] = set()

    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                study_ids.update(_extract_study_ids_from_json(value))
            if isinstance(value, str) and value.upper().startswith("ST"):
                study_ids.add(value.strip())
            if isinstance(key, str) and key.lower() in {"study_id", "studyid"}:
                if isinstance(value, str):
                    study_ids.add(value.strip())
    elif isinstance(obj, list):
        for item in obj:
            study_ids.update(_extract_study_ids_from_json(item))
    elif isinstance(obj, str):
        for token in obj.replace(",", " ").split():
            token = token.strip()
            if token.upper().startswith("ST") and len(token) >= 5:
                study_ids.add(token)

    return study_ids


def fetch_alzheimer_study_ids() -> List[str]:
    """Fetch study IDs related to Alzheimer disease from the MW API."""
    candidate_endpoints = [
        f"{MW_BASE_URL}/study/list/disease/Alzheimer",
        f"{MW_BASE_URL}/study/search/disease/Alzheimer",
        f"{MW_BASE_URL}/study/study_id/all",
    ]

    study_ids: Set[str] = set(FORCED_STUDY_IDS)

    headers = {"Accept": "application/json"}
    for endpoint in candidate_endpoints:
        try:
            response = requests.get(endpoint, timeout=60, headers=headers)
            if response.status_code != 200:
                continue
            content_type = response.headers.get("Content-Type", "")
            if "json" in content_type:
                data = response.json()
            else:
                # Try to parse plaintext or TSV-like responses.
                data = response.text
            study_ids.update(_extract_study_ids_from_json(data))
        except (requests.RequestException, json.JSONDecodeError):
            continue

    # Ensure forced IDs are present even if API calls failed.
    study_ids.update(FORCED_STUDY_IDS)

    # Sort for determinism.
    return sorted(study_ids)


def download_metabolomics_files(study_ids: Sequence[str]) -> None:
    """Download datatable and mwTab files for each study ID."""
    for study_id in study_ids:
        if not study_id:
            continue
        study_dir = MW_DIR / study_id
        study_dir.mkdir(parents=True, exist_ok=True)

        endpoints = {
            f"{study_id}_datatable.txt": f"{MW_BASE_URL}/study/study_id/{study_id}/datatable",
            f"{study_id}_mwtab.txt": f"{MW_BASE_URL}/study/study_id/{study_id}/mwtab",
        }

        for filename, url in endpoints.items():
            dest_path = study_dir / filename
            try:
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                dest_path.write_bytes(response.content)
                print(f"Saved {dest_path.relative_to(Path('.'))}")
            except requests.RequestException as exc:
                print(f"Failed to download {url}: {exc}", file=sys.stderr)


def download_ena_metadata(limit: int = 200) -> None:
    """Retrieve ENA microbiome run metadata related to Alzheimer disease."""
    params = {
        "result": "read_run",
        "query": "alzheimer AND (microbiome OR gut OR fecal OR stool)",
        "fields": "accession,study_accession,scientific_name,library_strategy,instrument_platform,collection_date",
        "limit": str(limit),
        "format": "tsv",
    }

    try:
        response = requests.get(ENA_BASE_URL, params=params, timeout=120)
        response.raise_for_status()
        ena_tsv_path = ENA_DIR / "ena_alz_microbiome_runs.tsv"
        ena_tsv_path.write_text(response.text)
        print(f"Saved {ena_tsv_path.relative_to(Path('.'))}")

        # Generate FASTQ URL stub file.
        fastq_lines: List[str] = []
        for line in response.text.splitlines()[1:]:  # skip header
            if not line.strip():
                continue
            run_accession = line.split("\t", 1)[0].strip()
            if run_accession:
                fastq_lines.append(
                    f"{run_accession}\thttps://www.ebi.ac.uk/ena/browser/api/fastq/{run_accession}"
                )
        ena_fastq_path = ENA_DIR / "ena_fastq_urls.txt"
        ena_fastq_path.write_text("\n".join(fastq_lines) + ("\n" if fastq_lines else ""))
        print(f"Saved {ena_fastq_path.relative_to(Path('.'))}")
    except requests.RequestException as exc:
        print(f"Failed to fetch ENA metadata: {exc}", file=sys.stderr)


def main() -> None:
    ensure_directories()
    study_ids = fetch_alzheimer_study_ids()
    if not study_ids:
        print("Warning: No study IDs found; proceeding with forced IDs only.", file=sys.stderr)
        study_ids = sorted(FORCED_STUDY_IDS)
    print(f"Found {len(study_ids)} Metabolomics Workbench study IDs.")
    download_metabolomics_files(study_ids)
    download_ena_metadata()


if __name__ == "__main__":
    main()
