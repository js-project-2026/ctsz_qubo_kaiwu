"""Parse 10x Genomics Cell Ranger matrix files (MTX + TSV)."""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path

from scipy.io import mmread
from scipy.sparse import csr_matrix, spmatrix


@dataclass
class TenxFeatures:
    ids: list[str]
    names: list[str]
    types: list[str]


@dataclass
class TenxMatrix:
    """Sparse count matrix as stored by Cell Ranger: features x cells."""

    counts: spmatrix
    barcodes: list[str]
    features: TenxFeatures


def _open_text(path: Path):
    path = Path(path)
    if path.suffix == ".gz" or str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return path.open("rt")


def parse_barcodes(path: Path | str) -> list[str]:
    with _open_text(Path(path)) as handle:
        return [line.strip() for line in handle if line.strip()]


def parse_features(path: Path | str) -> TenxFeatures:
    ids, names, types = [], [], []
    with _open_text(Path(path)) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            ids.append(parts[0])
            names.append(parts[1] if len(parts) > 1 else parts[0])
            types.append(parts[2] if len(parts) > 2 else "Gene Expression")
    return TenxFeatures(ids=ids, names=names, types=types)


def parse_feature_reference(path: Path | str) -> list[dict[str, str]]:
    with _open_text(Path(path)) as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def parse_mtx(path: Path | str) -> spmatrix:
    """Read a Matrix Market file (optionally .gz). Returns features x cells CSR."""
    matrix = mmread(str(path))
    return csr_matrix(matrix)


def parse_10x_directory(
    data_dir: Path | str,
    matrix_name: str | None = None,
) -> TenxMatrix:
    """
    Load a Cell Ranger filtered (or raw) feature-barcode directory.

    Expects files like:
      *_filtered_matrix.mtx.gz
      *_filtered_barcodes.tsv.gz
      *_filtered_features.tsv.gz
    """
    data_dir = Path(data_dir)
    matrix_path = _resolve_10x_file(data_dir, "matrix.mtx", matrix_name)
    barcode_path = _resolve_10x_file(data_dir, "barcodes.tsv", matrix_name)
    feature_path = _resolve_10x_file(data_dir, "features.tsv", matrix_name)

    barcodes = parse_barcodes(barcode_path)
    features = parse_features(feature_path)
    counts = parse_mtx(matrix_path)

    n_features, n_cells = counts.shape
    if n_features != len(features.ids):
        raise ValueError(
            f"MTX has {n_features} feature rows but features TSV has {len(features.ids)}"
        )
    if n_cells != len(barcodes):
        raise ValueError(
            f"MTX has {n_cells} cell columns but barcodes TSV has {len(barcodes)}"
        )
    return TenxMatrix(counts=counts, barcodes=barcodes, features=features)


def _resolve_10x_file(
    data_dir: Path, suffix: str, matrix_name: str | None
) -> Path:
    """Find barcodes/features/matrix, preferring filtered files."""
    data_dir = Path(data_dir)
    if matrix_name:
        stem = matrix_name.replace("_matrix.mtx.gz", "").replace("_matrix.mtx", "")
        candidates = [
            data_dir / f"{stem}_{suffix}.gz",
            data_dir / f"{stem}_{suffix}",
        ]
    else:
        candidates = []
        for prefix in ("*_filtered_", "*_raw_", "*_"):
            candidates.extend(data_dir.glob(f"{prefix}{suffix}.gz"))
            candidates.extend(data_dir.glob(f"{prefix}{suffix}"))
        # Also match files that are exactly matrix.mtx.gz
        candidates.extend(data_dir.glob(f"{suffix}.gz"))
        candidates.extend(data_dir.glob(suffix))

    existing = []
    seen = set()
    for path in candidates:
        path = Path(path)
        if path.exists() and path.resolve() not in seen:
            seen.add(path.resolve())
            existing.append(path)

    if not existing:
        raise FileNotFoundError(
            f"No *{suffix}[.gz] found in {data_dir}"
        )

    def _rank(path: Path) -> tuple[int, str]:
        name = path.name
        if "_filtered_" in name:
            return (0, name)
        if "_raw_" in name:
            return (2, name)
        return (1, name)

    existing.sort(key=_rank)
    return existing[0]
