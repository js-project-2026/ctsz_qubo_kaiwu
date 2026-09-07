"""Filesystem paths from environment variables. No machine-specific defaults.

Set these before loading data (shell export, or at the top of qubo_v1.py / qubo.ipynb):

  QUBO_REPO_DIR   Directory that contains qubo_model.py (defaults to cwd).
  QUBO_DATA_DIR   Directory that contains the GEO GSE308682 10x files listed below.

Test data (GEO GSE308682; not in git). Put all four files in QUBO_DATA_DIR:

  GSE308682_filtered_matrix.mtx.gz
  GSE308682_filtered_features.tsv.gz
  GSE308682_filtered_barcodes.tsv.gz
  GSE308682_feature_reference.csv.gz
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_REPO_DIR = "QUBO_REPO_DIR"
ENV_DATA_DIR = "QUBO_DATA_DIR"

# GEO GSE308682 Cell Ranger exports used by this project.
GSE308682_MATRIX = "GSE308682_filtered_matrix.mtx.gz"
GSE308682_FEATURES = "GSE308682_filtered_features.tsv.gz"
GSE308682_BARCODES = "GSE308682_filtered_barcodes.tsv.gz"
GSE308682_FEATURE_REFERENCE = "GSE308682_feature_reference.csv.gz"

GSE308682_REQUIRED_FILES = (
    GSE308682_MATRIX,
    GSE308682_FEATURES,
    GSE308682_BARCODES,
)
GSE308682_OPTIONAL_FILES = (GSE308682_FEATURE_REFERENCE,)


def repo_dir() -> Path:
    raw = os.environ.get(ENV_REPO_DIR, "").strip()
    path = Path(raw).expanduser() if raw else Path.cwd()
    return path.resolve()


def data_dir(explicit: Path | str | None = None) -> Path:
    """Resolve QUBO_DATA_DIR. Raises if unset or the 10x count files are missing."""
    if explicit is not None and str(explicit).strip():
        path = Path(explicit).expanduser().resolve()
    else:
        raw = os.environ.get(ENV_DATA_DIR, "").strip()
        if not raw:
            names = "\n".join(f"    {n}" for n in GSE308682_REQUIRED_FILES)
            extra = "\n".join(f"    {n}" for n in GSE308682_OPTIONAL_FILES)
            raise EnvironmentError(
                f"Set {ENV_DATA_DIR} to the folder that contains GEO GSE308682 10x files:\n"
                f"{names}\n"
                f"  optional CRISPR reference:\n{extra}\n"
                f"Example:\n  export {ENV_DATA_DIR}=/path/to/gse308682_dir\n"
                f"Or assign os.environ[{ENV_DATA_DIR!r}] at the top of qubo_v1.py / qubo.ipynb."
            )
        path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"{ENV_DATA_DIR} is not a directory: {path}"
        )
    missing = [name for name in GSE308682_REQUIRED_FILES if not (path / name).exists()]
    if missing:
        found = ", ".join(sorted(p.name for p in path.iterdir() if p.is_file())) or "(empty)"
        raise FileNotFoundError(
            f"{path} is missing required GSE308682 files: {', '.join(missing)}\n"
            f"Expected:\n"
            + "\n".join(f"  {n}" for n in GSE308682_REQUIRED_FILES)
            + f"\n  optional: {GSE308682_FEATURE_REFERENCE}\n"
            f"Found: {found}"
        )
    return path
