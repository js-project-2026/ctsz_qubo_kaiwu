"""Experiment helpers shared by qubo.ipynb and qubo_v1.py."""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, lasso_path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from data_loader import DEFAULT_N_TOP_GENES, load_scrna_qubo_data
from qubo_model import generate_synthetic_data


def load_experiment(
    use_real_data=True,
    data_dir=None,
    target_gene="RUNX1",
    n_top_genes=DEFAULT_N_TOP_GENES,
    n_samples=10000,
    n_features=50,
    target_mode="pseudotime",
    root_gene="HBE1",
):
    """Return X, y, true_features, feature_names (names is None for synthetic).

    Real data uses ``QUBO_DATA_DIR`` unless ``data_dir`` is passed. Expected
    GSE308682 files: GSE308682_filtered_matrix.mtx.gz,
    GSE308682_filtered_features.tsv.gz, GSE308682_filtered_barcodes.tsv.gz,
    GSE308682_feature_reference.csv.gz.
    """
    if use_real_data:
        X, y, true_features, feature_names = load_scrna_qubo_data(
            data_dir,
            target_gene=target_gene,
            n_top_genes=n_top_genes,
            target_mode=target_mode,
            root_gene=root_gene,
        )
        print(f"Real data (Pearson residuals): X={X.shape}, y={y.shape}")
        print(f"Feature names (first 10): {feature_names[:10]}")
        return X, y, true_features, feature_names
    X, y, true_features = generate_synthetic_data(
        n_samples=n_samples, n_features=n_features
    )
    print(f"Synthetic data: X={X.shape}, y={y.shape}")
    print(f"Source feature indices: {true_features}")
    return X, y, true_features, None


def target_cardinality(n_features, k=50):
    if n_features >= k:
        return k
    return max(5, n_features // 5)


def _lasso_top_k(X, y, K):
    """glmnet-style: pick the path alpha whose n_nonzero is closest to K.

    Fixed Lasso(alpha=0.01) on Pearson residuals is barely sparse, so
    argsort(|coef|)[-K:] was a contiguous HVG-index block of near-zeros.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - X.mean(axis=0)) / std
    alphas, coefs, _ = lasso_path(Xs, y, n_alphas=200)
    n_nz = np.sum(np.abs(coefs) > 1e-10, axis=0)
    path_i = int(np.argmin(np.abs(n_nz.astype(float) - K)))
    coef = coefs[:, path_i]
    nz = np.where(np.abs(coef) > 1e-10)[0]
    print(
        f"LASSO path: α={alphas[path_i]:.4g}, n_nonzero={n_nz[path_i]} "
        f"(target K={K})"
    )
    if nz.size == 0:
        return np.argsort(np.abs(coef))[-K:]
    if nz.size <= K:
        return nz
    return nz[np.argsort(np.abs(coef[nz]))[-K:]]


def compare_with_lasso_rfr(
    X,
    y,
    I,
    true_features,
    selected_idx_qubo,
    K=50,
    feature_names=None,
):
    selected_idx_lasso = _lasso_top_k(X, y, K)

    print("Fitting random forest (100 trees)...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    selected_idx_rf = np.argsort(rf_importance)[-K:]

    def recall_at_k(selected_idx):
        if not len(true_features):
            return float("nan")
        return len(set(selected_idx) & set(true_features)) / len(true_features)

    qset, lset, rset = set(map(int, selected_idx_qubo)), set(map(int, selected_idx_lasso)), set(map(int, selected_idx_rf))
    print(f"\n=== Feature selection (K={K}) ===")
    if len(true_features):
        print(f"Planted sources:     {true_features}")
    print(f"QUBO:                {sorted(map(int, selected_idx_qubo))}")
    print(f"LASSO:               {sorted(map(int, selected_idx_lasso))}")
    print(f"Random forest:       {sorted(map(int, selected_idx_rf))}")
    print(
        f"Overlaps: Q∩L={len(qset & lset)}/{K}, Q∩RF={len(qset & rset)}/{K}, "
        f"L∩RF={len(lset & rset)}/{K}, all3={len(qset & lset & rset)}"
    )
    if feature_names is not None:
        print(f"QUBO genes:          {[feature_names[i] for i in sorted(selected_idx_qubo)]}")
        print(f"LASSO genes:         {[feature_names[i] for i in sorted(selected_idx_lasso)]}")
        print(f"RF genes:            {[feature_names[i] for i in sorted(selected_idx_rf)]}")

    if len(true_features):
        print("\nRecall of planted sources:")
        print(f"  QUBO:           {recall_at_k(selected_idx_qubo):.2%}")
        print(f"  LASSO:          {recall_at_k(selected_idx_lasso):.2%}")
        print(f"  Random forest:  {recall_at_k(selected_idx_rf):.2%}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    colors = ["red" if i in selected_idx_qubo else "lightgray" for i in range(len(I))]
    axes[0].bar(range(len(I)), I, color=colors)
    axes[0].set_title("QUBO-selected features (red) vs MI importance")
    axes[0].set_ylabel("I(x; T)")
    viz = np.zeros(X.shape[1])
    viz[selected_idx_lasso] = 1.0
    colors = ["red" if i in selected_idx_lasso else "lightgray" for i in range(len(viz))]
    axes[1].bar(range(len(viz)), viz, color=colors)
    axes[1].set_title("LASSO-selected features (red); bar=1 if in top-K path set")
    axes[1].set_ylabel("selected")
    colors = ["red" if i in selected_idx_rf else "lightgray" for i in range(len(rf_importance))]
    axes[2].bar(range(len(rf_importance)), rf_importance, color=colors)
    axes[2].set_title("Random forest-selected features (red)")
    axes[2].set_xlabel("Feature index")
    axes[2].set_ylabel("Importance")
    plt.tight_layout()
    plt.show()
    return selected_idx_lasso, selected_idx_rf


def evaluate_selected_features(X, y, selected_indices):
    """Test MSE of a linear probe on the selected columns.

    OLS on Pearson residuals is ill-conditioned (collinear HVGs, fat tails),
    so sklearn's ``X @ coef_`` overflows. Z-score on the train split, then
    Ridge(α=1).
    """
    if len(selected_indices) == 0:
        return float("nan")
    X_selected = np.asarray(X[:, selected_indices], dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.3, random_state=42
    )
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-12] = 1.0
    X_train = (X_train - mu) / std
    X_test = (X_test - mu) / std
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        pred = np.asarray(model.predict(X_test), dtype=np.float64)
    if not np.all(np.isfinite(pred)):
        return float("nan")
    return float(mean_squared_error(y_test, pred))


def print_regression_mse(
    X,
    y,
    selected_idx_qubo,
    selected_idx_lasso,
    selected_idx_rf,
    extra=None,
):
    print("Regression MSE (z-scored Ridge α=1):")
    print(
        f"  QUBO Ocean tabu (k={len(selected_idx_qubo)}): "
        f"{evaluate_selected_features(X, y, selected_idx_qubo):.4f}"
    )
    extra = extra or {}
    for name, idx in extra.items():
        idx = np.asarray(idx)
        print(
            f"  {name} (k={len(idx)}): "
            f"{evaluate_selected_features(X, y, idx):.4f}"
        )
    print(
        f"  LASSO (k={len(selected_idx_lasso)}): "
        f"{evaluate_selected_features(X, y, selected_idx_lasso):.4f}"
    )
    print(
        f"  RF (k={len(selected_idx_rf)}): "
        f"{evaluate_selected_features(X, y, selected_idx_rf):.4f}"
    )
    print(
        f"  All features: "
        f"{evaluate_selected_features(X, y, list(range(X.shape[1]))):.4f}"
    )


def compare_kaiwu_tabu_on_same_q(
    Q,
    ocean_vec,
    ocean_energy,
    k,
    feature_names=None,
    num_reads=8,
    seed=0,
):
    """Solve the same Q with Kaiwu Tabu (local CPU). No CIM key required.

    Returns selected indices, or None if Kaiwu is missing / the solve fails.
    """
    from qubo_model import HAS_KAIWU, diagnose_qubo_solution, solve_qubo

    print("\n=== Ocean tabu vs Kaiwu tabu (same Q, no CIM key) ===")
    if not HAS_KAIWU:
        print(
            "Skip: kaiwu SDK not installed. "
            "pip install kaiwu==1.3.1  (official wheel is Python 3.10). "
            "kaiwu_tabu is classical CPU and does not need KAIWU_USER_ID."
        )
        return None
    print(
        f"Kaiwu TabuSearchOptimizer on n={Q.shape[0]} "
        f"(KAIWU_TABU_MAX_ITER={os.environ.get('KAIWU_TABU_MAX_ITER', '2000')})"
    )
    try:
        vec, energy = solve_qubo(
            Q, num_reads=num_reads, seed=seed, solver="kaiwu_tabu"
        )
    except Exception as exc:
        print(f"Skip Kaiwu tabu: {type(exc).__name__}: {exc}")
        return None
    report = diagnose_qubo_solution(Q, vec, energy=energy, k=k)
    idx = np.where(np.asarray(vec) == 1)[0]
    oset = set(map(int, np.where(np.asarray(ocean_vec) == 1)[0]))
    kset = set(map(int, idx))
    print(
        f"Ocean tabu:  |F*|={len(oset)}  energy={float(ocean_energy):.4f}"
    )
    print(f"Kaiwu tabu:  |F*|={len(kset)}  energy={float(energy):.4f}")
    print(f"Overlap Ocean∩Kaiwu={len(oset & kset)}/{k}")
    if feature_names is not None:
        print("Kaiwu tabu genes:", [feature_names[i] for i in sorted(idx)])
    if not report["accepted"]:
        print("Kaiwu tabu did not pass energy/cardinality acceptance.")
    return idx
