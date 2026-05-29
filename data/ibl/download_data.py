"""Downloading IBL subjectTrials tables.

This repo's preprocessing scripts expect parquet tables to exist under:
    data/ibl/tables_new/

Using ONE.load_aggregate without specifying a revision/tag can fail for some
subjects (e.g., "No default revision"). Instead, we download the specific
release tag directly from AWS.
"""

from pathlib import Path

from one.api import ONE

from data_utils import download_subjectTrials


if __name__ == "__main__":
    tag = "2023_Q1_Mohammadi_et_al"
    target_path = Path(__file__).resolve().parent / "tables_new"
    target_path.mkdir(parents=True, exist_ok=True)

    # Please use the Alyx password or other information if needed from 'one_params' at this link:
    # https://int-brain-lab.github.io/iblenv/_modules/oneibl/params.html
    # For example, the Alyx password is international.
    one = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international")

    out_paths = download_subjectTrials(one, target_path=target_path, tag=tag, overwrite=False, check_updates=True)
    print(f"Downloaded {len(out_paths)} parquet tables into: {target_path}")

