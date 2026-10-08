"""GSE198531 microglia QUBO with the developmental-age preprocessing fixes.

Run from the repo root with the same tabu environment as the previous run.
This script does not submit CIM.

    export QUBO_GSE198531_DIR=/path/to/dir
    export QUBO_TABU_WALLCLOCK_S=120
    export QUBO_TABU_NUM_RESTARTS=0
    python gse198531_qubo.py

The directory must contain:

    GSE198531_mus_microglia_tpm.txt
    GSE198531_series_matrix.txt

Steps before the existing solver:

1. Drop mt-, cytosolic ribosomal (Rpl/Rps), and Gm predicted genes.
   Highly variable genes are the top variance of log1p(TPM).
2. Map the six ages onto ranks 0..5 and scale them to [0, 1].
   Samples labeled A2M (inflammation, not age) are left out.
3. Mutual information uses 6 bins, one per age.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TPM_NAME = "GSE198531_mus_microglia_tpm.txt"
SERIES_NAME = "GSE198531_series_matrix.txt"

# Equal-rank order. Numeric day gaps are not used.
AGE_ORDER = ("E12.5", "E16.5", "E18.5", "P7", "P14", "Adult")
AGE_RANK = {name: i for i, name in enumerate(AGE_ORDER)}

N_TOP_GENES = 2500
MIN_CELLS = 10
N_BINS = 6
K_TARGET = 50
SOLVER = "tabu"
COMPARE_KAIWU_TABU = False

_RMB = re.compile(r"RMB\d+", re.IGNORECASE)
_MT = re.compile(r"^mt-", re.IGNORECASE)
_RIBO = re.compile(r"^Rp[ls]\d", re.IGNORECASE)
_GM = re.compile(r"^Gm\d+$", re.IGNORECASE)
_AGE = re.compile(r"\b(E12\.5|E16\.5|E18\.5|P7|P14|Adult)\b", re.IGNORECASE)
_A2M = re.compile(r"\bA2M\b", re.IGNORECASE)


def data_dir() -> Path:
    raw = os.environ.get("QUBO_GSE198531_DIR", "").strip()
    if not raw:
        raise SystemExit("Set QUBO_GSE198531_DIR to the folder with the GEO files.")
    path = Path(raw)
    missing = [name for name in (TPM_NAME, SERIES_NAME) if not (path / name).is_file()]
    if missing:
        raise SystemExit(f"Missing in {path}: {', '.join(missing)}")
    return path


def _split_geo_row(line: str) -> list[str]:
    parts = line.rstrip("\n").split("\t")
    return [part.strip().strip('"') for part in parts]


def technical_gene(name: str) -> str | None:
    """Return which filter drops this gene, or None to keep it."""
    symbol = name.split("|", 1)[-1].strip()
    if _MT.match(symbol):
        return "mt"
    if _RIBO.match(symbol):
        return "ribo"
    if _GM.match(symbol):
        return "Gm"
    return None


def age_label(text: str) -> str | None:
    """Developmental age token, or None. A2M is inflammation, not an age."""
    if _A2M.search(text) and _AGE.search(text) is None:
        return None
    match = _AGE.search(text)
    if match is None:
        return None
    token = match.group(1)
    if token.lower() == "adult":
        return "Adult"
    return token[0].upper() + token[1:]


def parse_series_matrix(path: Path) -> dict[str, str]:
    """Map RMB id to one of the six ages. A2M-only samples are omitted."""
    titles: list[str] = []
    characteristics: list[list[str]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Sample_title"):
                titles = _split_geo_row(line)[1:]
            elif line.startswith("!Sample_characteristics_ch1"):
                characteristics.append(_split_geo_row(line)[1:])
    if not titles:
        raise SystemExit(f"No !Sample_title rows in {path}")
    by_rmb: dict[str, str] = {}
    n_a2m = 0
    n_unparsed = 0
    width = len(titles)
    for index in range(width):
        chunks = [titles[index]]
        chunks.extend(row[index] for row in characteristics if index < len(row))
        text = " ".join(chunks)
        if _A2M.search(text) and _AGE.search(text) is None:
            n_a2m += 1
            continue
        label = age_label(text)
        rmb = _RMB.search(text)
        if label is None or rmb is None:
            n_unparsed += 1
            continue
        by_rmb[rmb.group(0).upper()] = label
    print(
        f"Series matrix: {len(by_rmb)} aged samples, "
        f"A2M excluded={n_a2m}, unmatched={n_unparsed}"
    )
    return by_rmb


def load_tpm(path: Path):
    import pandas as pd

    table = pd.read_csv(path, sep="\t", index_col=0)
    genes = np.asarray(table.index.astype(str))
    columns = np.asarray(table.columns.astype(str))
    values = np.asarray(table.to_numpy(), dtype=np.float64)
    print(f"TPM: {values.shape[0]} genes x {values.shape[1]} cells")
    return genes, columns, values


def align_cells(columns: np.ndarray, values: np.ndarray, age_by_rmb: dict[str, str]):
    keep = []
    ages = []
    missing = 0
    for col, name in enumerate(columns):
        rmb = _RMB.search(name)
        if rmb is None:
            missing += 1
            continue
        label = age_by_rmb.get(rmb.group(0).upper())
        if label is None:
            missing += 1
            continue
        keep.append(col)
        ages.append(label)
    if not keep:
        raise SystemExit("No TPM columns matched an aged RMB sample.")
    print(f"Aligned cells: {len(keep)} (TPM columns without an age: {missing})")
    counts = {name: ages.count(name) for name in AGE_ORDER}
    print("Age counts:", ", ".join(f"{name}={counts[name]}" for name in AGE_ORDER))
    return values[:, keep], np.asarray(ages)


def age_ranks(labels: np.ndarray) -> np.ndarray:
    """Ranks 0..5 scaled to [0, 1]. Adult is the last rank, not day 80."""
    ranks = np.asarray([AGE_RANK[label] for label in labels], dtype=np.float64)
    return ranks / float(len(AGE_ORDER) - 1)


def filter_genes(genes: np.ndarray, matrix: np.ndarray, min_cells: int):
    """Drop technical genes, then genes seen in fewer than min_cells."""
    reasons = {"mt": 0, "ribo": 0, "Gm": 0}
    keep = []
    for index, name in enumerate(genes):
        reason = technical_gene(name)
        if reason is not None:
            reasons[reason] += 1
            continue
        keep.append(index)
    kept_genes = genes[keep]
    kept = matrix[keep]
    detected = np.sum(kept > 0, axis=1)
    detect_mask = detected >= min_cells
    print(
        "Gene filter: "
        f"mt={reasons['mt']} ribo={reasons['ribo']} Gm={reasons['Gm']} "
        f"low_detection={int((~detect_mask).sum())} "
        f"remaining={int(detect_mask.sum())}"
    )
    return kept_genes[detect_mask], kept[detect_mask]


def log1p_hvg(genes: np.ndarray, matrix: np.ndarray, n_top: int):
    """Top genes by variance of log1p(TPM). matrix is genes x cells."""
    logged = np.log1p(matrix)
    variance = logged.var(axis=1)
    n_top = min(n_top, variance.size)
    order = np.argsort(variance)[-n_top:]
    order = order[np.argsort(variance[order])[::-1]]
    chosen = genes[order]
    print(f"Top {min(10, n_top)} HVG (log1p TPM variance): {list(chosen[:10])}")
    # Cells x genes, matching the QUBO loaders.
    return chosen, logged[order].T.copy()


def main():
    from qubo_experiment import (
        compare_with_lasso_rfr,
        print_regression_mse,
        target_cardinality,
    )
    from qubo_model import compute_mutual_information_matrix, solve_qubo_target_k

    folder = data_dir()
    genes, columns, tpm = load_tpm(folder / TPM_NAME)
    ages = parse_series_matrix(folder / SERIES_NAME)
    aligned, labels = align_cells(columns, tpm, ages)
    genes, aligned = filter_genes(genes, aligned, MIN_CELLS)
    feature_names, X = log1p_hvg(genes, aligned, N_TOP_GENES)
    y = age_ranks(labels)
    print(f"X={X.shape}  T in [{y.min():.3f}, {y.max():.3f}]  n_bins={N_BINS}")

    I, R = compute_mutual_information_matrix(X, y, n_bins=N_BINS)
    print(f"Importance I (first 5): {I[:5]}")
    k = target_cardinality(X.shape[1], k=K_TARGET)
    print(f"Target cardinality K={k}")
    selected, energy, alpha, Q, report = solve_qubo_target_k(I, R, k=k, solver=SOLVER)
    selected_idx = np.where(selected == 1)[0]
    print(f"α*={alpha:.4f}, energy={energy:.4f}, |F*|={len(selected_idx)}")
    print("QUBO selected genes:", [feature_names[i] for i in selected_idx])

    if COMPARE_KAIWU_TABU:
        from qubo_experiment import compare_kaiwu_tabu_on_same_q

        compare_kaiwu_tabu_on_same_q(
            Q, selected, energy, k=k, feature_names=feature_names
        )

    lasso_idx, rf_idx = compare_with_lasso_rfr(
        X,
        y,
        I,
        [],
        selected_idx,
        K=k,
        feature_names=feature_names,
    )
    print_regression_mse(X, y, selected_idx, lasso_idx, rf_idx)
    if not report["accepted"]:
        print("QUBO did not pass energy/cardinality acceptance.")


if __name__ == "__main__":
    main()
