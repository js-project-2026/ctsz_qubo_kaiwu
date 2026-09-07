# QUBO feature selection — Romero et al., Quantum Mach. Intell. (2025) 7:114
# Thin script: all logic lives in qubo_model.py, data_loader.py, qubo_experiment.py.

import numpy as np

from qubo_model import (
    compute_mutual_information_matrix,
    solve_qubo_target_k,
    solver_banner,
    available_solvers,
)
from qubo_experiment import (
    compare_with_lasso_rfr,
    load_experiment,
    print_regression_mse,
    target_cardinality,
)

# Paper-like real-data settings.
# Solvers: tabu | sa | leap | custom_sa | kaiwu_sa | kaiwu_tabu | kaiwu_cim
# leap needs DWAVE_API_TOKEN. kaiwu_cim needs KAIWU_USER_ID + KAIWU_SDK_CODE.
USE_REAL_DATA = True
N_TOP_GENES = 5000
TARGET_MODE = "pseudotime"
SOLVER = "tabu"

print(solver_banner(SOLVER))
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

selected_features_qubo, energy, alpha, Q = solve_qubo_target_k(
    I, R, k=K, solver=SOLVER
)
selected_idx_qubo = np.where(selected_features_qubo == 1)[0]
print(f"α*={alpha:.4f}, energy={energy:.4f}, |F*|={len(selected_idx_qubo)}")
print(f"QUBO selected indices: {selected_idx_qubo}")
if feature_names is not None:
    print("QUBO selected genes:", [feature_names[i] for i in selected_idx_qubo])

k_compare = max(len(selected_idx_qubo), 1)
selected_idx_lasso, selected_idx_rf = compare_with_lasso_rfr(
    X, y, I, true_features, selected_idx_qubo, K=k_compare, feature_names=feature_names
)
print_regression_mse(X, y, selected_idx_qubo, selected_idx_lasso, selected_idx_rf)
