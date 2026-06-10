"""Shared Kaggle download helper.

Used by fetch_rf and fetch_vision. Needs a free Kaggle API token:

  1. kaggle.com -> Account -> 'Create New API Token' -> downloads kaggle.json
  2. Put it at  %USERPROFILE%\\.kaggle\\kaggle.json
     (or set env KAGGLE_USERNAME / KAGGLE_KEY)

Then:  pip install kaggle   (already in requirements-core.txt)
"""
from __future__ import annotations
import os, sys, zipfile


def _have_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    cfg = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    return os.path.exists(cfg)


def download_dataset(slug: str, dest: str, unzip: bool = True) -> bool:
    """Download a Kaggle dataset (owner/name) into `dest`. Returns success."""
    os.makedirs(dest, exist_ok=True)
    if not _have_credentials():
        print(f"""
[kaggle] No credentials found - cannot download '{slug}'.
  1. kaggle.com -> Account -> Create New API Token  (saves kaggle.json)
  2. Move it to: {os.path.join(os.path.expanduser('~'), '.kaggle', 'kaggle.json')}
  3. Re-run this command.
(The pipeline still works on mock data without this.)
""")
        return False
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        print(f"[kaggle] downloading {slug} -> {dest} ...")
        api.dataset_download_files(slug, path=dest, unzip=unzip, quiet=False)
        print("[kaggle] done.")
        return True
    except Exception as e:
        print(f"[kaggle] failed: {type(e).__name__}: {e}")
        return False
