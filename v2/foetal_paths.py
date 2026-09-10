"""Paths for the Ranzoni / Cvejic fetal hematopoiesis QUBO (v2).

Does not touch the v1 GSE308682 10x layout. Set:

  QUBO_V2_DATA_DIR   GitLab repo root, or ``Data/ScanpyObjets``, or any folder
                     that contains the h5ad files listed below.

Expected Scanpy objects (from
https://gitlab.com/cvejic-group/integrative-scrna-scatac-human-foetal ):

  MergedAllSamples.h5ad            post-QC STAR counts (cells × genes)
  MergedAllSamples_PAGA.h5ad       hematopoietic cells + published dpt
  MergedAllSamples_annotated.h5ad  optional; cell-type labels if PAGA missing
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_DATA_DIR = "QUBO_V2_DATA_DIR"

H5AD_COUNTS = "MergedAllSamples.h5ad"
H5AD_PAGA = "MergedAllSamples_PAGA.h5ad"
H5AD_ANNOTATED = "MergedAllSamples_annotated.h5ad"
H5AD_BEFORE_QC = "MergedAllSamplesBeforeQC.h5ad"

REQUIRED_H5AD = (H5AD_COUNTS, H5AD_PAGA)


def _candidate_object_dirs(root: Path) -> list[Path]:
    return [
        root,
        root / "Data" / "ScanpyObjets",
        root / "Data" / "ScanpyObjects",
        root / "ScanpyObjets",
    ]


def scanpy_objects_dir(explicit: Path | str | None = None) -> Path:
    """Directory that contains ``MergedAllSamples.h5ad`` and the PAGA object."""
    if explicit is not None and str(explicit).strip():
        root = Path(explicit).expanduser().resolve()
    else:
        raw = os.environ.get(ENV_DATA_DIR, "").strip()
        if not raw:
            raise EnvironmentError(
                f"Set {ENV_DATA_DIR} to the Cvejic GitLab clone (or Data/ScanpyObjets).\n"
                f"Expected files: {', '.join(REQUIRED_H5AD)}\n"
                f"Example:\n  export {ENV_DATA_DIR}=/path/to/integrative-scrna-scatac-human-foetal"
            )
        root = Path(raw).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"{ENV_DATA_DIR} does not exist: {root}")

    for cand in _candidate_object_dirs(root):
        if cand.is_dir() and all((cand / name).is_file() for name in REQUIRED_H5AD):
            return cand

    searched = "\n".join(f"  {p}" for p in _candidate_object_dirs(root))
    raise FileNotFoundError(
        f"Could not find {REQUIRED_H5AD[0]} and {REQUIRED_H5AD[1]} under {root}.\n"
        f"Looked in:\n{searched}\n"
        "Clone https://gitlab.com/cvejic-group/integrative-scrna-scatac-human-foetal "
        "and point QUBO_V2_DATA_DIR at that repo (or its Data/ScanpyObjets folder)."
    )


def h5ad_path(name: str, data_dir: Path | str | None = None) -> Path:
    path = scanpy_objects_dir(data_dir) / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing Scanpy object: {path}")
    return path
