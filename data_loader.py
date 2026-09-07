"""Load GSE308682 (or any 10x directory) as QUBO feature-selection X, y.

Set QUBO_DATA_DIR to the folder that contains:

  GSE308682_filtered_matrix.mtx.gz
  GSE308682_filtered_features.tsv.gz
  GSE308682_filtered_barcodes.tsv.gz
  GSE308682_feature_reference.csv.gz   (optional; CRISPR target names)

Processing follows Romero et al. Methods §4.1: library-size / mito / detection
QC, analytic Pearson residuals (Lause et al. 2021), then a highly variable
gene pool. The continuous target T is either a held-out gene residual or
diffusion pseudotime (scanpy DPT), matching the paper's "cellular continuous
variable" rather than a CRISPR gene proxy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, issparse

from tqdm import tqdm
from tenx_parser import parse_10x_directory, parse_feature_reference
from paths import data_dir as resolve_data_dir

GENE_EXPRESSION = "Gene Expression"
DEFAULT_CRISPR_TARGETS = ("RUNX1", "MYB", "TCF3", "LMO2", "LDB1", "FLI1", "GATA2")
# Paper real-data pools were ~5,000 and ~9,661 processed genes.
DEFAULT_N_TOP_GENES = 5000


def load_scrna_qubo_data(
    data_dir: Path | str | None = None,
    target_gene: str = "RUNX1",
    n_top_genes: int | None = DEFAULT_N_TOP_GENES,
    min_counts: int = 1000,
    max_mito_frac: float = 0.15,
    min_genes_per_cell: int = 500,
    min_cells: int = 15,
    include_guide_features: bool = False,
    target_mode: str = "pseudotime",
    root_gene: str = "HBE1",
):
    """
    All cells that pass QC. Genes: detection filter, then top-N by variance
    of Pearson residuals.

    Parameters
    ----------
    target_mode
        ``"pseudotime"`` — y is scanpy diffusion pseudotime (paper-like T).
        ``"gene"`` — y is the Pearson residual of ``target_gene`` (held out of X).
    n_top_genes
        Paper used ~5,000 already-processed genes. Pairwise MI is O(p²); 5,000
        genes is the replication setting.
    root_gene
        For DPT, the cell with the *minimum* residual of this gene is iroot
        (late erythroid marker → more stem-like root on this dataset).

    Returns
    -------
    X, y, true_features, feature_names
        true_features is empty for real data (no planted sources).
    """
    data_dir = resolve_data_dir(data_dir)
    bundle = parse_10x_directory(data_dir)
    counts = bundle.counts.tocsr()
    feat_names = np.array(bundle.features.names)
    feat_ids = np.array(bundle.features.ids)
    feat_types = np.array(bundle.features.types)

    if include_guide_features:
        type_mask = np.ones(len(feat_types), dtype=bool)
    else:
        type_mask = feat_types == GENE_EXPRESSION

    gene_ids = feat_ids[type_mask]
    gene_names = feat_names[type_mask]
    gene_counts = counts[type_mask, :]

    X_cells = gene_counts.T.tocsr()
    lib = np.asarray(X_cells.sum(axis=1)).ravel()
    n_genes_cell = np.asarray((X_cells > 0).sum(axis=1)).ravel()
    mito = np.char.startswith(gene_names.astype(str), "MT-")
    if mito.any():
        mito_frac = np.asarray(X_cells[:, mito].sum(axis=1)).ravel() / np.maximum(lib, 1.0)
    else:
        mito_frac = np.zeros(X_cells.shape[0])

    cell_keep = (
        (lib >= min_counts)
        & (n_genes_cell >= min_genes_per_cell)
        & (mito_frac <= max_mito_frac)
    )
    gene_counts = gene_counts[:, cell_keep]

    n_detected = np.asarray((gene_counts > 0).sum(axis=1)).ravel()
    gene_keep = n_detected >= min_cells
    gene_counts = gene_counts[gene_keep, :]
    gene_names = gene_names[gene_keep]
    gene_ids = gene_ids[gene_keep]

    target_mode = target_mode.strip().lower()
    if target_mode not in {"gene", "pseudotime"}:
        raise ValueError("target_mode must be 'gene' or 'pseudotime'")

    hold_out_target = target_mode == "gene"
    target_idx = None
    if hold_out_target:
        target_idx = _find_gene_index(gene_names, gene_ids, target_gene)

    n_keep = n_top_genes
    if hold_out_target and n_keep is not None:
        n_keep = n_top_genes + 1
    if n_keep is not None and gene_counts.shape[0] > n_keep:
        var = pearson_residual_variance(gene_counts, theta=100)
        if hold_out_target:
            var[target_idx] = np.inf
        order = np.argsort(var)[::-1][:n_keep]
        if hold_out_target and target_idx not in order:
            order = np.append(order[:-1], target_idx)
        gene_counts = gene_counts[order, :]
        gene_names = gene_names[order]
        gene_ids = gene_ids[order]
        if hold_out_target:
            target_idx = _find_gene_index(gene_names, gene_ids, target_gene)

    print("Computing Pearson residuals...")
    residuals = compute_pearson_residual(gene_counts, theta=100)
    residuals = np.asarray(residuals)
    X_all = residuals.T.astype(np.float64)
    names = gene_names.astype(str)

    if target_mode == "gene":
        y = X_all[:, target_idx].astype(np.float64)
        feat_mask = np.ones(X_all.shape[1], dtype=bool)
        feat_mask[target_idx] = False
        X = X_all[:, feat_mask]
        names = names[feat_mask]
        y_label = f"Pearson residual of {target_gene}"
    else:
        y, iroot = diffusion_pseudotime(X_all, names, root_gene=root_gene)
        X = X_all
        y_label = f"DPT (iroot=argmin {root_gene} residual, cell {iroot})"

    y = np.asarray(y, dtype=np.float64).ravel()
    finite = np.isfinite(y)
    if not finite.all():
        y = y.copy()
        y[~finite] = np.nanmedian(y[finite]) if finite.any() else 0.0
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)
    print(f"Target T: {y_label}")
    return X, y, [], names.tolist()


def pearson_residual_variance(matrix, theta=100, batch_genes=400):
    """Per-gene variance of clipped Pearson residuals without a full dense matrix."""
    if not issparse(matrix):
        matrix = csr_matrix(matrix)
    n_genes, n_cells = matrix.shape
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    col_sums = np.asarray(matrix.sum(axis=0)).ravel()
    total = float(matrix.sum())
    clip = np.sqrt(n_cells)
    var = np.empty(n_genes, dtype=np.float64)
    starts = range(0, n_genes, batch_genes)
    for start in tqdm(starts, desc="HVG residual variance", dynamic_ncols=True):
        sl = slice(start, min(start + batch_genes, n_genes))
        block = matrix[sl].toarray().astype(np.float64)
        expected = np.outer(row_sums[sl], col_sums) / max(total, 1.0)
        std = np.sqrt(expected + (expected ** 2) / theta)
        resid = (block - expected) / np.maximum(std, 1e-12)
        resid = np.clip(np.nan_to_num(resid, nan=0.0, posinf=0.0, neginf=0.0), -clip, clip)
        var[sl] = resid.var(axis=1)
    return var


def compute_pearson_residual(matrix, theta=100):
    """Analytic Pearson residuals, features x cells (Lause et al. 2021)."""
    if not issparse(matrix):
        matrix = csr_matrix(matrix)
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    col_sums = np.asarray(matrix.sum(axis=0)).ravel()
    total = float(matrix.sum())
    expected = (row_sums[:, None] * col_sums[None, :]) / max(total, 1.0)
    std = np.sqrt(expected + (expected ** 2) / theta)
    dense = matrix.toarray().astype(np.float64)
    residuals = (dense - expected) / np.maximum(std, 1e-12)
    residuals = np.nan_to_num(residuals, nan=0.0, posinf=0.0, neginf=0.0)
    clip = np.sqrt(matrix.shape[1])
    return np.clip(residuals, -clip, clip)


def diffusion_pseudotime(X, gene_names, root_gene="HBE1", n_pcs=30, n_neighbors=30):
    """scanpy DPT on Pearson-residual cells × genes (paper: T = pseudotime)."""
    import warnings

    # tqdm.auto emits IProgress warnings in Jupyter without ipywidgets.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="IProgress not found")
        import scanpy as sc

    adata = sc.AnnData(X)
    adata.var_names = np.asarray(gene_names).astype(str)
    adata.var_names_make_unique()
    n_comps = min(n_pcs, X.shape[1] - 1, X.shape[0] - 1)
    names = np.array(adata.var_names)
    if root_gene in names:
        iroot = int(np.argmin(np.asarray(X[:, names == root_gene]).ravel()))
    else:
        iroot = int(np.argmin(X.sum(axis=1)))
    adata.uns["iroot"] = iroot

    bar = tqdm(total=4, desc="DPT", dynamic_ncols=True)
    bar.set_postfix_str("PCA")
    sc.tl.pca(adata, n_comps=n_comps)
    bar.update(1)
    bar.set_postfix_str("neighbors")
    sc.pp.neighbors(
        adata, n_neighbors=n_neighbors, n_pcs=min(n_pcs, adata.obsm["X_pca"].shape[1])
    )
    bar.update(1)
    bar.set_postfix_str("diffmap")
    sc.tl.diffmap(adata)
    bar.update(1)
    bar.set_postfix_str("dpt")
    sc.tl.dpt(adata)
    bar.update(1)
    bar.close()
    y = np.asarray(adata.obs["dpt_pseudotime"].to_numpy(), dtype=np.float64)
    return y, iroot


def load_guide_assignments(data_dir: Path | str | None = None):
    data_dir = resolve_data_dir(data_dir)
    bundle = parse_10x_directory(data_dir)
    types = np.array(bundle.features.types)
    names = np.array(bundle.features.names)
    guide_mask = types == "CRISPR Guide Capture"
    guide_counts = bundle.counts[guide_mask, :].T.tocsr()
    return guide_counts, names[guide_mask].tolist(), bundle.barcodes


def _find_gene_index(names: np.ndarray, ids: np.ndarray, query: str) -> int:
    query = query.strip()
    for arr in (names, ids):
        hits = np.where(arr == query)[0]
        if len(hits):
            return int(hits[0])
    raise KeyError(f"Target gene {query!r} not found in Gene Expression features")


def list_crispr_targets(data_dir: Path | str | None = None) -> list[str]:
    ref_files = list(resolve_data_dir(data_dir).glob("*feature_reference.csv*"))
    if not ref_files:
        return list(DEFAULT_CRISPR_TARGETS)
    rows = parse_feature_reference(ref_files[0])
    names = []
    for row in rows:
        name = row.get("target_gene_name") or row.get("name")
        if name and name not in names:
            names.append(name)
    return names
