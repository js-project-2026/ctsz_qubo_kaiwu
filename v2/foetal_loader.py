"""Load Ranzoni et al. fetal liver/BM Smart-seq2 as QUBO X, y.

Default (benchmark):
  - cells = hematopoietic set in ``MergedAllSamples_PAGA.h5ad`` (endothelium dropped)
  - counts from ``MergedAllSamples.h5ad``
  - T = published ``dpt_pseudotime`` (PAGA + ForceAtlas2, HSC/MPP root)
  - features = log1p(CPM) then top-N HVGs (not the authors' 1,000-HVG autoencoder)

``transform="pearson"`` is a sensitivity check: Lause residuals are a UMI model
and are not the Smart-seq2 protocol used in the paper.

Parent ``data_loader.py`` (GSE308682 / 10x) is left unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, issparse
from tqdm import tqdm

from foetal_paths import H5AD_COUNTS, H5AD_PAGA, h5ad_path


def pearson_residual_variance(matrix, theta=100, batch_genes=400):
    """Per-gene variance of clipped Pearson residuals (same formula as v1)."""
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

DEFAULT_N_TOP_GENES = 5000
HSC_CLUSTER = "HSC-MPPs"
DEFAULT_ROOT_GENE = "MLLT3"
EXCLUDE_CLUSTERS = {"Endothelial"}


def load_foetal_qubo_data(
    data_dir: Path | str | None = None,
    n_top_genes: int | None = DEFAULT_N_TOP_GENES,
    transform: str = "log",
    target_mode: str = "paga_dpt",
    target_gene: str = DEFAULT_ROOT_GENE,
    root_gene: str = DEFAULT_ROOT_GENE,
    drop_unspecified: bool = False,
    n_cells: int | None = None,
    cell_sample_seed: int = 0,
    min_cells: int = 10,
):
    """Return X, y, true_features, feature_names, cell_meta.

    Parameters
    ----------
    transform
        ``"log"`` — library-size normalize to 10,000 then log1p (Ranzoni-like).
        ``"pearson"`` — analytic Pearson residuals (Lause 2021; UMI-oriented).
    target_mode
        ``"paga_dpt"`` — published PAGA diffusion pseudotime (recommended).
        ``"dpt"`` — recompute scanpy DPT on the QUBO feature matrix, iroot =
        max ``root_gene`` among HSC/MPP cells.
        ``"gene"`` — hold out ``target_gene`` from X; y is that gene's values.
    n_cells
        If set, subsample this many cells after hematopoietic filtering (smoke test).
    """
    import warnings

    import anndata as ad

    transform = transform.strip().lower()
    target_mode = target_mode.strip().lower()
    if transform not in {"log", "pearson"}:
        raise ValueError("transform must be 'log' or 'pearson'")
    if target_mode not in {"paga_dpt", "dpt", "gene"}:
        raise ValueError("target_mode must be 'paga_dpt', 'dpt', or 'gene'")

    counts_path = h5ad_path(H5AD_COUNTS, data_dir)
    paga_path = h5ad_path(H5AD_PAGA, data_dir)
    print(f"Loading counts: {counts_path}")
    print(f"Loading PAGA / dpt: {paga_path}")
    # Published PAGA h5ad stores neighbor graphs in .uns; current anndata
    # migrates them to .obsp and emits a FutureWarning on read.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Moving element from \.uns\['neighbors'\]",
            category=FutureWarning,
        )
        counts_ad = ad.read_h5ad(counts_path)
        paga_ad = ad.read_h5ad(paga_path)

    keep_cells, cluster, origin, sample, dpt = _hematopoietic_index(
        counts_ad, paga_ad, drop_unspecified=drop_unspecified
    )
    counts = _as_dense_counts(counts_ad[keep_cells].X)
    names = np.asarray(counts_ad.var_names).astype(str)
    ensembl = None
    if "Ensembl" in counts_ad.var.columns:
        ensembl = np.asarray(counts_ad.var["Ensembl"]).astype(str)
    print(
        f"Hematopoietic cells: {counts.shape[0]} × {counts.shape[1]} genes "
        f"(PAGA n={paga_ad.n_obs})"
    )

    if n_cells is not None and n_cells < counts.shape[0]:
        rng = np.random.default_rng(cell_sample_seed)
        pick = np.sort(rng.choice(counts.shape[0], size=int(n_cells), replace=False))
        counts = counts[pick]
        cluster = cluster[pick]
        origin = origin[pick]
        sample = sample[pick]
        dpt = dpt[pick]
        print(f"Smoke subsample: {counts.shape[0]} cells (seed={cell_sample_seed})")

    n_detected = (counts > 0).sum(axis=0)
    gene_keep = n_detected >= min_cells
    counts = counts[:, gene_keep]
    names = names[gene_keep]
    if ensembl is not None:
        ensembl = ensembl[gene_keep]
    names, counts = _make_unique_names(names, counts)
    print(f"After min_cells={min_cells}: {counts.shape[0]} cells × {counts.shape[1]} genes")

    hold_out = target_mode == "gene"
    target_idx = None
    if hold_out:
        target_idx = _find_gene_index(names, ensembl, target_gene)

    n_keep = n_top_genes
    if hold_out and n_keep is not None:
        n_keep = n_top_genes + 1
    if n_keep is not None and counts.shape[1] > n_keep:
        if transform == "pearson":
            print("HVG by Pearson-residual variance...")
            var = pearson_residual_variance(counts.T, theta=100)
        else:
            print("HVG by log1p(CPM) variance...")
            var = _log_cpm_variance(counts)
        if hold_out:
            var[target_idx] = np.inf
        order = np.argsort(var)[::-1][:n_keep]
        if hold_out and target_idx not in order:
            order = np.append(order[:-1], target_idx)
        counts = counts[:, order]
        names = names[order]
        if ensembl is not None:
            ensembl = ensembl[order]
        if hold_out:
            target_idx = _find_gene_index(names, ensembl, target_gene)

    if transform == "pearson":
        print("Computing Pearson residuals...")
        X_all = compute_pearson_residual(counts.T, theta=100).T.astype(np.float64)
        x_label = "Pearson residuals"
    else:
        print("Computing log1p(CPM)...")
        X_all = _log_cpm(counts)
        x_label = "log1p(CPM)"

    if target_mode == "gene":
        y = X_all[:, target_idx].astype(np.float64)
        feat_mask = np.ones(X_all.shape[1], dtype=bool)
        feat_mask[target_idx] = False
        X = X_all[:, feat_mask]
        names = names[feat_mask]
        y_label = f"{x_label} of held-out {target_gene}"
    elif target_mode == "paga_dpt":
        y = np.asarray(dpt, dtype=np.float64)
        X = X_all
        y_label = "published PAGA dpt_pseudotime (HSC/MPP root)"
    else:
        y, iroot = _dpt_hsc_root(
            X_all, names, cluster, hsc_cluster=HSC_CLUSTER, root_gene=root_gene
        )
        X = X_all
        y_label = (
            f"recomputed DPT (iroot=argmax {root_gene} in {HSC_CLUSTER}, cell {iroot})"
        )

    y = np.asarray(y, dtype=np.float64).ravel()
    finite = np.isfinite(y)
    if not finite.all():
        y = y.copy()
        y[~finite] = np.nanmedian(y[finite]) if finite.any() else 0.0
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)

    meta = {
        "cluster": np.asarray(cluster).astype(str),
        "origin": np.asarray(origin).astype(str),
        "sample": np.asarray(sample).astype(str),
        "transform": transform,
        "target_mode": target_mode,
        "x_label": x_label,
        "y_label": y_label,
        "n_cells_full_paga": int(paga_ad.n_obs),
    }
    print(f"Target T: {y_label}")
    print(f"X={X.shape} ({x_label}), clusters={_count_map(cluster)}")
    return X, y, [], [str(n) for n in names], meta


def _hematopoietic_index(counts_ad, paga_ad, drop_unspecified=False):
    """Align PAGA hematopoietic cells onto the count object."""
    cluster_col = _first_present(paga_ad.obs, ("Cluster", "annotation", "manual_annotation"))
    if cluster_col is None:
        raise KeyError("PAGA object has no Cluster / annotation column")
    cluster = paga_ad.obs[cluster_col].astype(str)
    drop = set(EXCLUDE_CLUSTERS)
    if drop_unspecified:
        drop.add("Unspecified")
    keep_names = [
        name
        for name, lab in zip(paga_ad.obs_names, cluster)
        if lab not in drop and name in counts_ad.obs_names
    ]
    if not keep_names:
        raise RuntimeError("No overlapping hematopoietic cells between counts and PAGA h5ad")
    missing = paga_ad.n_obs - len(keep_names)
    if missing:
        print(f"Dropped {missing} PAGA cells (endothelium / unspecified / not in counts)")

    loc = counts_ad.obs_names.get_indexer(keep_names)
    if np.any(loc < 0):
        raise RuntimeError("Internal error: PAGA cell not found in counts object")
    paga_sub = paga_ad[keep_names]
    dpt = paga_sub.obs["dpt_pseudotime"].to_numpy() if "dpt_pseudotime" in paga_sub.obs else np.full(len(keep_names), np.nan)
    origin = (
        paga_sub.obs["origin"].astype(str).to_numpy()
        if "origin" in paga_sub.obs
        else np.array(["unknown"] * len(keep_names))
    )
    sample = (
        paga_sub.obs["sample"].astype(str).to_numpy()
        if "sample" in paga_sub.obs
        else np.array(["unknown"] * len(keep_names))
    )
    return loc, paga_sub.obs[cluster_col].astype(str).to_numpy(), origin, sample, dpt


def _as_dense_counts(X):
    if issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


def _log_cpm(counts, scale=10000.0):
    lib = counts.sum(axis=1, keepdims=True)
    lib[lib <= 0] = 1.0
    return np.log1p(counts / lib * scale).astype(np.float64)


def _log_cpm_variance(counts, batch_genes=800):
    n_cells, n_genes = counts.shape
    var = np.empty(n_genes, dtype=np.float64)
    lib = counts.sum(axis=1, keepdims=True)
    lib[lib <= 0] = 1.0
    scale = 10000.0
    for start in tqdm(range(0, n_genes, batch_genes), desc="HVG log-CPM variance", dynamic_ncols=True):
        sl = slice(start, min(start + batch_genes, n_genes))
        block = np.log1p(counts[:, sl] / lib * scale)
        var[sl] = block.var(axis=0)
    return var


def _dpt_hsc_root(X, gene_names, cluster_labels, hsc_cluster=HSC_CLUSTER, root_gene=DEFAULT_ROOT_GENE):
    """scanpy DPT with iroot = max root_gene among HSC/MPP cells."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="IProgress not found")
        import scanpy as sc

    names = np.asarray(gene_names).astype(str)
    clusters = np.asarray(cluster_labels).astype(str)
    if root_gene in names:
        g = np.asarray(X[:, names == root_gene]).ravel()
        hsc = clusters == hsc_cluster
        if hsc.any():
            local = np.where(hsc)[0]
            iroot = int(local[int(np.argmax(g[hsc]))])
        else:
            iroot = int(np.argmax(g))
    else:
        iroot = 0
        print(f"root_gene {root_gene!r} missing; iroot=0")

    adata = sc.AnnData(X)
    adata.var_names = names
    adata.var_names_make_unique()
    adata.uns["iroot"] = iroot
    n_comps = min(30, X.shape[1] - 1, X.shape[0] - 1)
    bar = tqdm(total=4, desc="DPT", dynamic_ncols=True)
    bar.set_postfix_str("PCA")
    sc.tl.pca(adata, n_comps=n_comps)
    bar.update(1)
    bar.set_postfix_str("neighbors")
    sc.pp.neighbors(adata, n_neighbors=30, n_pcs=min(30, adata.obsm["X_pca"].shape[1]))
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


def _make_unique_names(names, counts):
    seen = {}
    out = []
    for n in names:
        if n not in seen:
            seen[n] = 0
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}-{seen[n]}")
    return np.asarray(out, dtype=object), counts


def _find_gene_index(names, ensembl, query: str) -> int:
    query = query.strip()
    hits = np.where(np.asarray(names).astype(str) == query)[0]
    if len(hits):
        return int(hits[0])
    if ensembl is not None:
        hits = np.where(np.asarray(ensembl).astype(str) == query)[0]
        if len(hits):
            return int(hits[0])
    raise KeyError(f"Target gene {query!r} not found in fetal matrix")


def _first_present(obs, columns):
    for c in columns:
        if c in obs.columns:
            return c
    return None


def _count_map(labels, n=8):
    labels = np.asarray(labels).astype(str)
    vals, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    parts = [f"{vals[i]}={counts[i]}" for i in order[:n]]
    if order.size > n:
        parts.append("...")
    return ", ".join(parts)


__all__ = [
    "DEFAULT_N_TOP_GENES",
    "HSC_CLUSTER",
    "load_foetal_qubo_data",
]
