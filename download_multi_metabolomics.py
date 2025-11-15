"""Download multiple Metabolomics Workbench studies used in the multi-cohort
Alzheimer's metabolomics pipeline.

The ISEF project trains models on a panel of human metabolomics cohorts focused
on the tryptophan (TRP) and short-chain fatty acid (SCFA) pathways.  This script
ensures that raw exports for each study are present under ``data/raw`` so that
``process_multi_metabolomics.py`` can parse and harmonise them.

The script fetches Metabolomics Workbench studies using their REST download
endpoints.  For ST000046 and ST000047 we simply check that the directories exist
because those files were provided with the repository.  For the newer cohorts
(ST000462, ST001152, ST001050) we retrieve the mwtab and/or datatable text
exports and write them into the appropriate ``data/raw/<study_id>/`` folder.

Run from the repository root:

.. code-block:: bash

   python download_multi_metabolomics.py

"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import requests

RAW_DATA_DIR = Path("data/raw")
STUDIES_WITH_EXISTING_DATA: Tuple[str, ...] = ("ST000046", "ST000047")
STUDIES_TO_DOWNLOAD: Tuple[str, ...] = ("ST000462", "ST001152", "ST001050")
DOWNLOAD_TEMPLATES: Tuple[str, ...] = (
    "https://www.metabolomicsworkbench.org/rest/study/{study_id}/mwtab",
    "https://www.metabolomicsworkbench.org/rest/study/{study_id}/datatable",
)


def ensure_directories(study_ids: Iterable[str]) -> Dict[str, Path]:
    """Return a mapping of study ID to its raw data directory, creating if needed."""

    directories: Dict[str, Path] = {}
    for study_id in study_ids:
        study_dir = RAW_DATA_DIR / study_id
        study_dir.mkdir(parents=True, exist_ok=True)
        directories[study_id] = study_dir
    return directories


def download_study(study_id: str, study_dir: Path) -> None:
    """Download mwtab/datatable exports for ``study_id`` into ``study_dir``."""

    for template in DOWNLOAD_TEMPLATES:
        url = template.format(study_id=study_id)
        target_name = f"{study_id}_{url.rsplit('/', 1)[-1]}.txt"
        target_path = study_dir / target_name

        logging.info("Downloading %s to %s", url, target_path)
        response = requests.get(url, timeout=120)
        if response.status_code != 200:
            logging.warning("Failed to download %s (status %s)", url, response.status_code)
            continue

        text = response.text.strip()
        if not text:
            logging.warning("Response for %s was empty; skipping", url)
            continue

        target_path.write_text(text, encoding="utf-8")
        logging.info("Saved %s", target_path)


def main() -> None:
    """Entry point that prepares directories and downloads the requested studies."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ensure_directories(STUDIES_WITH_EXISTING_DATA + STUDIES_TO_DOWNLOAD)

    for study_id in STUDIES_WITH_EXISTING_DATA:
        study_dir = RAW_DATA_DIR / study_id
        if any(study_dir.iterdir()):
            logging.info("%s already present in %s; skipping download", study_id, study_dir)
        else:
            logging.warning("%s directory exists but is empty; please populate manually", study_id)

    for study_id in STUDIES_TO_DOWNLOAD:
        study_dir = RAW_DATA_DIR / study_id
        download_study(study_id, study_dir)


if __name__ == "__main__":
    main()
