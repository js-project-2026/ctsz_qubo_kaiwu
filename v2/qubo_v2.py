# QUBO feature selection v2 — Ranzoni et al. fetal hematopoiesis (Smart-seq2)
# Solver / MI / α-search: parent qubo_model.py (unchanged).

import os
import sys
from pathlib import Path

V2_DIR = Path(__file__).resolve().parent
REPO = V2_DIR.parent
os.environ.setdefault("QUBO_REPO_DIR", str(REPO))
os.environ.setdefault(
    "QUBO_V2_DATA_DIR",
    "/Users/jinshang/Projects/integrative-scrna-scatac-human-foetal",
)

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

import numpy as np

from qubo_model import (
    compute_mutual_information_matrix,
    solve_qubo_target_k,
    solver_banner,
    available_solvers,
)
import qubo_model as _qubo_model
from experiment import (
    compare_kaiwu_tabu_on_same_q,
    compare_with_lasso_rfr,
    load_experiment,
    print_regression_mse,
    target_cardinality,
)

# SMOKE=True: subsample cells/genes for a fast check. False: full 5k-HVG benchmark.
SMOKE = False
USE_REAL_DATA = True
N_TOP_GENES = 80 if SMOKE else 5000
N_CELLS = 200 if SMOKE else None
TRANSFORM = "log"          # log | pearson
TARGET_MODE = "paga_dpt"   # paga_dpt | dpt | gene
ROOT_GENE = "MLLT3"
SOLVER = "tabu"
COMPARE_KAIWU_TABU = not SMOKE
K_TARGET = 10 if SMOKE else 50

print(f"Kernel Python {sys.version.split()[0]}  (Kaiwu official wheel needs 3.10.x)")
if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"Need Python 3.10.x for Kaiwu (got {sys.version.split()[0]}). "
        "Use .venv-py310 / kernel Python 3.10 (Kaiwu)."
    )
print(solver_banner(SOLVER))
print("qubo_model loaded from", _qubo_model.__file__)
print("v2 foetal_loader from", V2_DIR)
print("Available solvers:", {k: v["installed"] for k, v in available_solvers().items()})
print(f"SMOKE={SMOKE}  n_top_genes={N_TOP_GENES}  n_cells={N_CELLS}")

X, y, true_features, feature_names, meta = load_experiment(
    use_real_data=USE_REAL_DATA,
    n_top_genes=N_TOP_GENES,
    transform=TRANSFORM,
    target_mode=TARGET_MODE,
    root_gene=ROOT_GENE,
    n_cells=N_CELLS,
)

I, R = compute_mutual_information_matrix(X, y, n_bins=10)
print(f"Importance I (first 5): {I[:5]}")
print(f"Redundancy R (5x5):\n{R[:5, :5]}")

K = target_cardinality(X.shape[1], k=K_TARGET)
print(f"Target cardinality K={K}")

selected_features_qubo, energy, alpha, Q, report = solve_qubo_target_k(
    I, R, k=K, solver=SOLVER
)
selected_idx_qubo = np.where(selected_features_qubo == 1)[0]
print(f"α*={alpha:.4f}, energy={energy:.4f}, |F*|={len(selected_idx_qubo)}")
print(f"QUBO selected indices: {selected_idx_qubo}")
if feature_names is not None:
    print("QUBO selected genes:", [feature_names[i] for i in selected_idx_qubo])

kaiwu_idx = None
if COMPARE_KAIWU_TABU:
    kaiwu_idx = compare_kaiwu_tabu_on_same_q(
        Q,
        selected_features_qubo,
        energy,
        k=K,
        feature_names=feature_names,
    )

k_compare = K
if not report["accepted"]:
    print(
        "QUBO did not pass energy/cardinality acceptance. "
        f"LASSO and RF are compared at target K={K}, not |F*|={len(selected_idx_qubo)}."
    )
selected_idx_lasso, selected_idx_rf = compare_with_lasso_rfr(
    X, y, I, true_features, selected_idx_qubo, K=k_compare, feature_names=feature_names
)
extra = {}
if kaiwu_idx is not None:
    extra["QUBO Kaiwu tabu"] = kaiwu_idx
print_regression_mse(
    X, y, selected_idx_qubo, selected_idx_lasso, selected_idx_rf, extra=extra
)
