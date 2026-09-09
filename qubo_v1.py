# QUBO feature selection — Romero et al., Quantum Mach. Intell. (2025) 7:114
# Thin script: all logic lives in qubo_model.py, data_loader.py, qubo_experiment.py.

import os
from pathlib import Path

# --- paths: set these before loading real data ---
# QUBO_REPO_DIR: folder that contains qubo_model.py (default: this file's directory).
# QUBO_DATA_DIR: folder with GEO GSE308682 10x files:
#   GSE308682_filtered_matrix.mtx.gz
#   GSE308682_filtered_features.tsv.gz
#   GSE308682_filtered_barcodes.tsv.gz
#   GSE308682_feature_reference.csv.gz
os.environ.setdefault("QUBO_REPO_DIR", str(Path(__file__).resolve().parent))
# Uncomment and edit, or: export QUBO_DATA_DIR=/path/to/gse308682_dir
# os.environ["QUBO_DATA_DIR"] = "/path/to/gse308682_dir"

import sys

_repo = Path(os.environ["QUBO_REPO_DIR"]).expanduser().resolve()
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import numpy as np

from qubo_model import (
    compute_mutual_information_matrix,
    require_python_310,
    solve_qubo_target_k,
    solver_banner,
    available_solvers,
)
import qubo_model as _qubo_model
from qubo_experiment import (
    compare_kaiwu_tabu_on_same_q,
    compare_with_lasso_rfr,
    load_experiment,
    print_regression_mse,
    target_cardinality,
)

# Paper-like real-data settings.
# Solvers: tabu | sa | leap | custom_sa | kaiwu_sa | kaiwu_tabu | kaiwu_cim
# leap needs DWAVE_API_TOKEN. kaiwu_cim needs KAIWU_USER_ID + KAIWU_SDK_CODE.
# kaiwu_tabu is local CPU and does not need a CIM key.
# tabu timeout: QUBO_TABU_TIMEOUT_MS (ms/read). Ocean's 20 ms default is not used.
USE_REAL_DATA = True
N_TOP_GENES = 5000
TARGET_MODE = "pseudotime"
SOLVER = "tabu"
COMPARE_KAIWU_TABU = True

require_python_310()
print(f"Kernel Python {sys.version.split()[0]}  (Kaiwu official wheel needs 3.10.x)")
print(solver_banner(SOLVER))
print("qubo_model loaded from", _qubo_model.__file__)
print("Available solvers:", {k: v["installed"] for k, v in available_solvers().items()})

X, y, true_features, feature_names = load_experiment(
    use_real_data=USE_REAL_DATA,
    n_top_genes=N_TOP_GENES,
    target_mode=TARGET_MODE,
)

I, R = compute_mutual_information_matrix(X, y, n_bins=10)
print(f"Importance I (first 5): {I[:5]}")
print(f"Redundancy R (5x5):\n{R[:5, :5]}")

K = target_cardinality(X.shape[1], k=50)
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
        f"LASSO and RF are compared at target K={K}, not |F*|={len(selected_idx_qubo)}. "
        "MSE below is not evidence that QUBO minimized Eq. (4)."
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
