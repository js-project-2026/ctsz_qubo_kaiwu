"""v2 experiment helpers. Solver / LASSO / RF live in the parent repo.

v2 modules are named ``foetal_loader`` / ``foetal_paths`` so they never shadow
parent ``data_loader`` / ``paths`` when the notebook cwd is ``v2/``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_V2 = Path(__file__).resolve().parent
_PARENT = _V2.parent


def _load_parent(mod_name: str, filename: str):
    """Load a repo-root module under its public name, replacing a v2 shadow."""
    path = _PARENT / filename
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

# Jupyter cwd is often v2/, which used to shadow these names.
_load_parent("paths", "paths.py")
_load_parent("data_loader", "data_loader.py")
_parent_exp = _load_parent("qubo_experiment", "qubo_experiment.py")

compare_kaiwu_tabu_on_same_q = _parent_exp.compare_kaiwu_tabu_on_same_q
compare_with_lasso_rfr = _parent_exp.compare_with_lasso_rfr
print_regression_mse = _parent_exp.print_regression_mse
target_cardinality = _parent_exp.target_cardinality

from qubo_model import generate_synthetic_data  # noqa: E402

if str(_V2) not in sys.path:
    sys.path.insert(0, str(_V2))
from foetal_loader import DEFAULT_N_TOP_GENES, load_foetal_qubo_data  # noqa: E402


def load_experiment(
    use_real_data=True,
    data_dir=None,
    n_top_genes=DEFAULT_N_TOP_GENES,
    n_samples=10000,
    n_features=50,
    transform="log",
    target_mode="paga_dpt",
    target_gene="MLLT3",
    root_gene="MLLT3",
    n_cells=None,
    cell_sample_seed=0,
    drop_unspecified=False,
):
    """Return X, y, true_features, feature_names, cell_meta (meta is {} for synthetic)."""
    if use_real_data:
        X, y, true_features, feature_names, meta = load_foetal_qubo_data(
            data_dir,
            n_top_genes=n_top_genes,
            transform=transform,
            target_mode=target_mode,
            target_gene=target_gene,
            root_gene=root_gene,
            n_cells=n_cells,
            cell_sample_seed=cell_sample_seed,
            drop_unspecified=drop_unspecified,
        )
        print(f"Fetal Smart-seq2 ({meta['x_label']}): X={X.shape}, y={y.shape}")
        print(f"Feature names (first 10): {feature_names[:10]}")
        return X, y, true_features, feature_names, meta
    X, y, true_features = generate_synthetic_data(
        n_samples=n_samples, n_features=n_features
    )
    print(f"Synthetic data: X={X.shape}, y={y.shape}")
    print(f"Source feature indices: {true_features}")
    return X, y, true_features, None, {}
