"""QUBO feature selection following Romero et al., Quantum Mach. Intell. (2025) 7:114.

Cost (Eq. 4):  Q(F) = -α Σ I_i F_i + (1-α) Σ R_ij F_i F_j
Matrix:        Q = (1-α) R - α diag(I)
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    from dwave.samplers import SimulatedAnnealingSampler, TabuSampler

    HAS_DWAVE = True
    HAS_TABU = True
except ImportError:
    HAS_DWAVE = False
    HAS_TABU = False
    SimulatedAnnealingSampler = None
    TabuSampler = None

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    njit = None

try:
    import kaiwu as _kaiwu

    HAS_KAIWU = True
except ImportError:
    HAS_KAIWU = False
    _kaiwu = None

try:
    import kaiwu.torch_plugin  # noqa: F401

    HAS_KAIWU_PLUGIN = True
except ImportError:
    HAS_KAIWU_PLUGIN = False

from tqdm import tqdm


def require_python_310():
    """Kaiwu official wheels are 3.10-only. Fail fast on other interpreters."""
    if sys.version_info[:2] != (3, 10):
        raise SystemExit(
            "Kaiwu official wheel needs Python 3.10.x, "
            f"got {sys.version.split()[0]}. "
            "Use .venv-py310 or the 'Python 3.10 (Kaiwu)' kernel."
        )


def _progress(iterable, **kwargs):
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("leave", True)
    return tqdm(iterable, **kwargs)


SOURCE_IDX = [5, 11, 7, 1, 14]
TARGET_IDX = [16, 17, 18, 19, 20]


def generate_synthetic_data(
    n_samples=10000,
    n_features=50,
    noise=0.1,
    rho=0.1,
    seed=42,
):
    """Synthetic data from Romero et al. §2.3 (n=50, p=10,000)."""
    rng = np.random.default_rng(seed)
    B = rng.normal(0, 1, (n_features, n_features))
    cov = B.T @ B
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    X = rng.multivariate_normal(np.zeros(n_features), corr, size=n_samples)

    source_idx = list(SOURCE_IDX)
    target_idx = list(TARGET_IDX)
    for s, t in zip(source_idx, target_idx):
        X[:, t] = X[:, s] + rho * rng.normal(0, 1, n_samples)

    s1, s2, s3, s4, s5 = source_idx
    y = (
        0.5 * np.cos(7 * X[:, s4])
        + np.sin(X[:, s3] * X[:, s2])
        + 0.1 * np.exp(X[:, s5]) * np.log2(np.abs(10 * X[:, s1]) + 1e-12)
        + noise * rng.normal(0, 1, n_samples)
    )
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)
    X = StandardScaler().fit_transform(X)
    return X, y, source_idx


def _quantile_bin_1d(x, n_bins=10):
    x = np.asarray(x, dtype=np.float64)
    if np.allclose(x, x[0]):
        return np.zeros(x.size, dtype=np.int32), 1
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        return np.zeros(x.size, dtype=np.int32), 1
    idx = np.clip(np.digitize(x, edges[1:-1], right=False), 0, edges.size - 2)
    return idx.astype(np.int32), int(edges.size - 1)


def _entropy_from_counts(counts):
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def discrete_mutual_information(a, b, ka, kb):
    joint = np.bincount(a * kb + b, minlength=ka * kb).reshape(ka, kb)
    return (
        _entropy_from_counts(joint.sum(axis=1))
        + _entropy_from_counts(joint.sum(axis=0))
        - _entropy_from_counts(joint)
    )


def compute_mutual_information_matrix(X, y, n_bins=10):
    """Quantile-binned discrete MI (paper Methods, Eq. 5–6)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n_samples, n_features = X.shape

    codes = np.zeros((n_features, n_samples), dtype=np.int32)
    nbins = np.ones(n_features, dtype=np.int32)
    for i in _progress(range(n_features), desc="Quantile-bin genes"):
        codes[i], nbins[i] = _quantile_bin_1d(X[:, i], n_bins=n_bins)
    y_code, y_bins = _quantile_bin_1d(y, n_bins=n_bins)

    I = np.zeros(n_features)
    for i in _progress(range(n_features), desc="Importance I(x; T)"):
        I[i] = discrete_mutual_information(codes[i], y_code, int(nbins[i]), y_bins)

    R = np.zeros((n_features, n_features))
    nbins_i = nbins.astype(np.int32)
    for i in _progress(range(n_features), desc="Redundancy R (pairwise MI)"):
        if HAS_NUMBA:
            row = _redundancy_row(codes, nbins_i, i)
            R[i, i + 1 :] = row[i + 1 :]
            R[i + 1 :, i] = row[i + 1 :]
        else:
            for j in range(i + 1, n_features):
                mi = discrete_mutual_information(
                    codes[i], codes[j], int(nbins[i]), int(nbins[j])
                )
                R[i, j] = mi
                R[j, i] = mi
    np.fill_diagonal(R, 0.0)
    return I, R


if HAS_NUMBA:

    @njit(cache=True)
    def _mi_codes(a, b, ka, kb):
        n = a.size
        joint = np.zeros(ka * kb, dtype=np.float64)
        for t in range(n):
            joint[a[t] * kb + b[t]] += 1.0
        ha = 0.0
        hb = 0.0
        hab = 0.0
        for i in range(ka):
            s = 0.0
            for j in range(kb):
                s += joint[i * kb + j]
            if s > 0.0:
                p = s / n
                ha -= p * np.log2(p)
        for j in range(kb):
            s = 0.0
            for i in range(ka):
                s += joint[i * kb + j]
            if s > 0.0:
                p = s / n
                hb -= p * np.log2(p)
        for k in range(ka * kb):
            if joint[k] > 0.0:
                p = joint[k] / n
                hab -= p * np.log2(p)
        return ha + hb - hab

    @njit(cache=True)
    def _redundancy_row(codes, nbins, i):
        p = codes.shape[0]
        row = np.zeros(p, dtype=np.float64)
        ka = int(nbins[i])
        for j in range(i + 1, p):
            row[j] = _mi_codes(codes[i], codes[j], ka, int(nbins[j]))
        return row


def build_qubo_matrix(I, R, alpha=0.5, k=None):
    """Eq. (4): Q = (1-α) R - α diag(I).

    If k is set, R is scaled by 1/(k-1) as in the authors' solver
    (qfeatures/annealing_functions.py) so unconstrained QUBO yields ~k bits.
    """
    I = np.asarray(I, dtype=np.float64).ravel()
    R = np.asarray(R, dtype=np.float64)
    R_use = R.copy()
    np.fill_diagonal(R_use, 0.0)
    if k is not None and k > 1:
        R_use = R_use / (k - 1)
    return (1.0 - alpha) * R_use - alpha * np.diag(I)


def _qubo_dict(Q, show_progress=False):
    """Dense dict for samplers that still want sample_qubo. Prefer _bqm_from_q."""
    qubo = {}
    n = Q.shape[0]
    rows = range(n)
    if show_progress and n >= 500:
        rows = _progress(rows, desc="Pack QUBO dict")
    for i in rows:
        qi = Q[i]
        for j in range(n):
            v = qi[j]
            if v != 0:
                qubo[(i, j)] = float(v)
    return qubo


def _bqm_from_q(Q):
    """Binary QUBO whose energy equals Fᵀ Q F (Q symmetrized)."""
    import dimod

    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix")
    Q = 0.5 * (Q + Q.T)
    return dimod.BinaryQuadraticModel(Q, "BINARY")


def _energy(Q, f):
    """Fᵀ Q F for a binary (or 0/1) mask. Non-finite results become +inf."""
    f = np.asarray(f, dtype=np.float64).ravel()
    Q = np.asarray(Q, dtype=np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        e = float(f @ Q @ f)
    if not np.isfinite(e):
        return float(np.inf)
    return e


def tabu_timeout_ms(n, explicit=None):
    """Milliseconds per Ocean tabu read. Ocean's default is 20 ms — too small at p~5000."""
    if explicit is not None:
        return max(1, int(explicit))
    raw = os.environ.get("QUBO_TABU_TIMEOUT_MS", "").strip()
    if raw:
        return max(1, int(raw))
    return int(max(500, min(30_000, 2 * int(n))))


def diagnose_qubo_solution(Q, vec, energy=None, k=None):
    """Acceptance checks: energy vs empty/singleton, and |F*| vs target k.

    A minimizer of FᵀQF must beat the empty mask (E=0) and the best singleton
    (E=min Q_ii). Positive energy is not an optimum. Returns a dict; prints a report.
    """
    Q = np.asarray(Q, dtype=np.float64)
    vec = np.asarray(vec, dtype=int).ravel()
    n = vec.size
    n_sel = int(vec.sum())
    e = float(energy) if energy is not None else _energy(Q, vec)
    e_re = _energy(Q, vec)
    diag = np.diag(Q)
    i_star = int(np.argmin(diag))
    e_one = float(diag[i_star])
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        qf = Q @ vec.astype(np.float64)
    qf = np.asarray(qf, dtype=np.float64)
    if np.all(np.isfinite(qf)):
        remove_delta = -2.0 * qf + diag
        add_delta = 2.0 * qf + diag
        n_better_off = int(np.sum((vec == 1) & (remove_delta < -1e-8)))
        n_better_on = int(np.sum((vec == 0) & (add_delta < -1e-8)))
    else:
        n_better_off = n_better_on = -1
    energy_ok = e <= min(0.0, e_one) + 1e-6
    k_tol = 0 if k is None else (0 if int(k) < 10 else max(2, int(k) // 10))
    k_ok = True if k is None else abs(n_sel - int(k)) <= k_tol
    accepted = bool(energy_ok and k_ok)
    print("QUBO acceptance:")
    print(f"  |F*|={n_sel}" + (f"  target k={int(k)}  tol={k_tol}" if k is not None else ""))
    print(f"  energy reported={e:.4f}  recomputed FᵀQF={e_re:.4f}")
    print(f"  empty mask energy=0  best singleton Q[{i_star},{i_star}]={e_one:.4f}")
    print(
        f"  one-bit moves that would lower E: turn_off={n_better_off}  turn_on={n_better_on}"
        + ("  (skipped; QF overflowed)" if n_better_off < 0 else "")
    )
    if abs(e - e_re) > 1e-4:
        print("  WARNING: reported energy disagrees with FᵀQF.")
    if e > 1e-8:
        print(
            "  FAIL: energy > 0, worse than selecting nothing. "
            "This is not a minimizer of Eq. (4). Do not use MSE to argue otherwise."
        )
    elif not energy_ok:
        print("  FAIL: energy is worse than the best singleton; local search did not finish.")
    if k is not None and not k_ok:
        print(f"  FAIL: |F*|={n_sel} is not within {k_tol} of target k={int(k)}.")
    if accepted:
        print("  PASS: energy beats empty/singleton and cardinality is near k.")
    else:
        print("  overall: NOT ACCEPTED as a k-gene QUBO panel.")
    return {
        "accepted": accepted,
        "energy_ok": energy_ok,
        "k_ok": k_ok,
        "n_sel": n_sel,
        "energy": e,
        "energy_recomputed": e_re,
        "energy_singleton": e_one,
        "n_better_off": n_better_off,
        "n_better_on": n_better_on,
        "n": n,
    }


def custom_simulated_annealing(
    Q,
    max_iter=20000,
    temp_init=5.0,
    temp_min=0.01,
    cooling_rate=0.995,
    seed=None,
):
    """Unconstrained bit-flip SA (minimize Fᵀ Q F). No fixed cardinality."""
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    # Start at the empty mask (E=0). Random 50% ones can sit at huge positive energy.
    current = np.zeros(n, dtype=int)
    current_energy = _energy(Q, current)
    best = current.copy()
    best_energy = current_energy
    temp = temp_init
    it = 0
    accepted = 0
    while temp > temp_min and it < max_iter:
        i = int(rng.integers(0, n))
        new = current.copy()
        new[i] = 1 - new[i]
        new_energy = _energy(Q, new)
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-12)):
            current = new
            current_energy = new_energy
            accepted += 1
            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy
        temp *= cooling_rate
        it += 1
    return best.astype(int), best_energy, accepted


def _sample_to_vec(sample, n):
    vec = np.zeros(n, dtype=int)
    for idx, val in sample.items():
        vec[int(idx)] = int(val)
    return vec


def _qubo_to_ising_matrix(Q):
    """Convert QUBO Q to the Ising matrix Kaiwu Optimizer.solve expects.

    Uses kaiwu.conversion (or kaiwu.qubo) when installed. QUBO linear terms
    become an extra auxiliary spin; decode with _ising_row_to_binary.
    """
    Q = np.asarray(Q, dtype=np.float64)
    Q = 0.5 * (Q + Q.T)
    if not HAS_KAIWU:
        raise ImportError(
            "Kaiwu solvers need the kaiwu SDK: pip install kaiwu==1.3.1"
        )
    conv = getattr(_kaiwu, "conversion", None)
    if conv is not None and hasattr(conv, "qubo_matrix_to_ising_matrix"):
        ising, bias = conv.qubo_matrix_to_ising_matrix(Q)
        return np.asarray(ising, dtype=np.float64), float(bias)
    qubo_mod = getattr(_kaiwu, "qubo", None)
    if qubo_mod is not None and hasattr(qubo_mod, "qubo_matrix_to_ising_matrix"):
        ising, bias = qubo_mod.qubo_matrix_to_ising_matrix(Q)
        return np.asarray(ising, dtype=np.float64), float(bias)
    raise ImportError(
        "kaiwu.conversion.qubo_matrix_to_ising_matrix is missing; upgrade kaiwu"
    )


def _ising_row_to_binary(spins, n_qubo):
    """Map a ±1 Ising row (possibly with one aux spin) to {0,1}^n_qubo."""
    s = np.asarray(spins, dtype=np.float64).ravel()
    s = np.where(s >= 0, 1.0, -1.0)
    if s.size == n_qubo + 1:
        s = s[:-1] * s[-1]
    elif s.size != n_qubo:
        raise ValueError(
            f"Ising sample length {s.size} does not match n={n_qubo} or n+1"
        )
    return ((s + 1.0) / 2.0).astype(int)


def _best_binary_from_ising_samples(samples, Q):
    n = Q.shape[0]
    rows = np.atleast_2d(np.asarray(samples))
    if rows.size == 0:
        raise RuntimeError("Kaiwu solver returned no samples")
    best_x = None
    best_e = np.inf
    for row in rows:
        x = _ising_row_to_binary(row, n)
        e = _energy(Q, x)
        if e < best_e:
            best_e = e
            best_x = x
    return best_x, float(best_e)


def _kaiwu_credentials():
    user = os.environ.get("KAIWU_USER_ID") or os.environ.get("USER_ID")
    code = (
        os.environ.get("KAIWU_SDK_CODE")
        or os.environ.get("KAIWU_SDK_TOKEN")
        or os.environ.get("SDK_CODE")
    )
    return user, code


def _init_kaiwu_license():
    """Optional. Classical Kaiwu solvers do not need CIM keys.

    If USER_ID / SDK_CODE are set, initialize a local license file. Do not
    call license.ensure_license() here — that can block on a console prompt.
    """
    if not HAS_KAIWU:
        return
    user, code = _kaiwu_credentials()
    lic = getattr(_kaiwu, "license", None)
    if lic is not None and user and code and hasattr(lic, "init"):
        lic.init(user, code)


def _kaiwu_optimizer(kind, num_reads, seed):
    if kind in {"kaiwu_sa", "kaiwu_tabu"}:
        _init_kaiwu_license()
    if kind == "kaiwu_sa":
        return _kaiwu.classical.SimulatedAnnealingOptimizer(
            size_limit=max(1, int(num_reads)),
            rand_seed=seed,
        )
    if kind == "kaiwu_tabu":
        max_iter = int(os.environ.get("KAIWU_TABU_MAX_ITER", "2000"))
        return _kaiwu.classical.TabuSearchOptimizer(
            max_iter=max_iter,
            size_limit=max(1, int(num_reads)),
        )
    if kind == "kaiwu_cim":
        user, code = _kaiwu_credentials()
        if not user or not code:
            raise RuntimeError(
                "kaiwu_cim needs KAIWU_USER_ID and KAIWU_SDK_CODE "
                "(from https://platform.qboson.com/)"
            )
        opt = _kaiwu.cim.CIMOptimizer(
            user_id=user,
            sdk_code=code,
            task_name=os.environ.get("KAIWU_TASK_NAME", "qubo_feature_selection"),
            wait=True,
        )
        reducer = getattr(_kaiwu.cim, "PrecisionReducer", None)
        if reducer is not None:
            opt = reducer(
                opt,
                precision=8,
                truncated_precision=10,
                only_feasible_solution=False,
            )
        return opt
    raise ValueError(f"Unknown Kaiwu solver {kind!r}")


def _solve_kaiwu(Q, kind, num_reads, seed):
    if not HAS_KAIWU:
        raise ImportError(
            "Install Kaiwu: pip install kaiwu==1.3.1 "
            "and optionally git+https://github.com/qboson/kaiwu-pytorch-plugin.git"
        )
    ising, _bias = _qubo_to_ising_matrix(Q)
    worker = _kaiwu_optimizer(kind, num_reads=num_reads, seed=seed)
    samples = worker.solve(ising)
    if samples is None:
        raise RuntimeError(
            "Kaiwu CIM returned None (task still running). Use wait=True / poll."
        )
    return _best_binary_from_ising_samples(samples, Q)


def available_solvers():
    """Installed backends. 'quantum' means cloud QPU / photonic CIM, not local SA."""
    return {
        "tabu": {"installed": HAS_TABU, "kind": "classical", "stack": "dwave-ocean-sdk"},
        "sa": {"installed": HAS_DWAVE, "kind": "classical", "stack": "dwave-ocean-sdk"},
        "leap": {"installed": HAS_DWAVE, "kind": "quantum-hybrid", "stack": "dwave-ocean-sdk"},
        "custom_sa": {"installed": True, "kind": "classical", "stack": "this repo"},
        "kaiwu_sa": {"installed": HAS_KAIWU, "kind": "classical", "stack": "kaiwu SDK"},
        "kaiwu_tabu": {"installed": HAS_KAIWU, "kind": "classical", "stack": "kaiwu SDK"},
        "kaiwu_cim": {
            "installed": HAS_KAIWU,
            "kind": "quantum",
            "stack": "kaiwu CIM (QBoson photonic); kaiwu-pytorch-plugin optional",
        },
    }


def solver_banner(solver=None):
    solver = (solver or _default_solver()).lower()
    labels = {
        "tabu": "D-Wave Ocean TabuSampler (classical, local; MATLAB tabu analog)",
        "sa": "D-Wave Ocean SimulatedAnnealingSampler (classical, local)",
        "leap": "D-Wave LeapHybridSampler (quantum-classical hybrid; DWAVE_API_TOKEN)",
        "custom_sa": "Built-in bit-flip simulated annealing (classical)",
        "kaiwu_sa": "Kaiwu SimulatedAnnealingOptimizer (classical CPU; no CIM key)",
        "kaiwu_tabu": "Kaiwu TabuSearchOptimizer (classical CPU; no CIM key)",
        "kaiwu_cim": "Kaiwu CIMOptimizer (QBoson photonic CIM; KAIWU_USER_ID + KAIWU_SDK_CODE)",
    }
    extra = ""
    if solver.startswith("kaiwu") and HAS_KAIWU_PLUGIN:
        extra = " [kaiwu.torch_plugin imported]"
    elif solver.startswith("kaiwu") and not HAS_KAIWU:
        extra = " [kaiwu not installed]"
    return labels.get(solver, f"Using solver={solver}") + extra


def _default_solver():
    """Paper used MATLAB tabu + Leap hybrid. Local Ocean tabu is the closest default."""
    if HAS_TABU:
        return "tabu"
    if HAS_DWAVE:
        return "sa"
    if HAS_KAIWU:
        return "kaiwu_sa"
    return "custom_sa"


def solve_qubo(
    Q,
    num_reads=8,
    seed=None,
    max_iter=20000,
    solver=None,
    timeout_ms=None,
):
    """Minimize QUBO. Switch backends with ``solver``.

    D-Wave Ocean (``dwave-ocean-sdk`` is already a dependency):
      ``tabu`` / ``sa`` — classical local samplers
      ``leap`` — Leap hybrid (needs DWAVE_API_TOKEN)

    Ocean TabuSampler default timeout is **20 milliseconds per read**. This
    function overrides that (see ``tabu_timeout_ms`` / ``QUBO_TABU_TIMEOUT_MS``)
    and seeds the first read at the all-zero mask so a timed-out search cannot
    return a random ~p/2 bitstring with energy worse than 0.

    Kaiwu / QBoson:
      ``kaiwu_sa`` / ``kaiwu_tabu`` — classical local CPU; no CIM key.
        Enterprise SDK may still want a one-time local license file.
      ``kaiwu_cim`` — photonic CIM (KAIWU_USER_ID + KAIWU_SDK_CODE)

    ``custom_sa`` — bit-flip SA in this file (also starts at all zeros).
    """
    solver = (solver or _default_solver()).lower()
    aliases = {"cim": "kaiwu_cim", "kaiwu": "kaiwu_cim", "dwave_sa": "sa", "ocean_tabu": "tabu"}
    solver = aliases.get(solver, solver)
    n = Q.shape[0]
    num_reads = max(1, int(num_reads))

    if solver in {"kaiwu_sa", "kaiwu_tabu", "kaiwu_cim"}:
        return _solve_kaiwu(Q, solver, num_reads=num_reads, seed=seed)

    if solver == "leap":
        from dwave.system import LeapHybridSampler

        token = os.environ.get("DWAVE_API_TOKEN")
        kwargs = {}
        if token:
            kwargs["token"] = token
        bqm = _bqm_from_q(Q)
        result = LeapHybridSampler(**kwargs).sample(bqm)
        return _sample_to_vec(result.first.sample, n), float(result.first.energy)

    if solver == "tabu":
        if not HAS_TABU:
            raise ImportError("TabuSampler requires dwave-ocean-sdk / dwave-samplers")
        to_ms = tabu_timeout_ms(n, timeout_ms)
        bqm = _bqm_from_q(Q)
        print(
            f"TabuSampler: n={n}, num_reads={num_reads}, timeout={to_ms} ms/read "
            f"(Ocean default is 20 ms), first initial state = all zeros"
        )
        result = TabuSampler().sample(
            bqm,
            num_reads=num_reads,
            seed=seed,
            timeout=to_ms,
            initial_states=np.zeros(n, dtype=np.int8),
            initial_states_generator="random",
        )
        vec = _sample_to_vec(result.first.sample, n)
        energy = float(result.first.energy)
        return vec, energy

    if solver == "sa":
        if not HAS_DWAVE:
            raise ImportError("SimulatedAnnealingSampler requires dwave-ocean-sdk")
        bqm = _bqm_from_q(Q)
        result = SimulatedAnnealingSampler().sample(
            bqm,
            num_reads=num_reads,
            seed=seed,
            initial_states=np.zeros(n, dtype=np.int8),
            initial_states_generator="random",
        )
        return _sample_to_vec(result.first.sample, n), float(result.first.energy)

    if solver != "custom_sa":
        known = ", ".join(sorted(available_solvers()))
        raise ValueError(f"Unknown solver {solver!r}. Choose one of: {known}")

    vec, energy, _ = custom_simulated_annealing(Q, max_iter=max_iter, seed=seed)
    return vec, energy


def _n_selected(
    I, R, alpha, k, num_reads=2, seed=0, solver=None, timeout_ms=None
):
    Q = build_qubo_matrix(I, R, alpha=alpha, k=k)
    vec, energy = solve_qubo(
        Q,
        num_reads=num_reads,
        seed=seed,
        max_iter=4000,
        solver=solver,
        timeout_ms=timeout_ms,
    )
    return int(vec.sum()), vec, float(energy)


def solve_qubo_target_k(
    I,
    R,
    k=50,
    num_reads=8,
    seed=0,
    n_bisect=16,
    solver=None,
    timeout_ms=None,
    bisect_reads=2,
):
    """Bisect α ∈ [0, 1] so the unconstrained solution has about k ones.

    Matches the authors' root_scalar search over α (annealing_functions.py).
    n_selected increases with α (α=0 → few/none, α=1 → all I>0 features).

    Logs every probe (α, |F*|, energy). ``best_alpha`` is the probe with
    smallest |n_sel−k|, not the last tqdm midpoint.

    Returns
    -------
    vec, energy, alpha, Q, report
        ``report`` is the dict from ``diagnose_qubo_solution``.
    """
    k = int(k)
    solver = solver or _default_solver()
    lo, hi = 0.0, 1.0
    best_vec = None
    best_alpha = 0.5
    best_diff = np.inf
    best_energy = np.inf
    probe_reads = max(1, int(bisect_reads))
    bar = _progress(range(n_bisect), desc="Bisect α for |F*|≈k")
    for step in bar:
        mid = 0.5 * (lo + hi)
        n_sel, vec, energy = _n_selected(
            I,
            R,
            mid,
            k,
            num_reads=probe_reads,
            seed=seed,
            solver=solver,
            timeout_ms=timeout_ms,
        )
        bar.set_postfix(
            alpha=f"{mid:.4f}", n_sel=n_sel, E=f"{energy:.2f}", refresh=False
        )
        print(
            f"  α-bisect {step + 1}/{n_bisect}: α={mid:.6f}  |F*|={n_sel}  "
            f"energy={energy:.4f}  |n-k|={abs(n_sel - k)}"
        )
        diff = abs(n_sel - k)
        if diff < best_diff or (diff == best_diff and energy < best_energy):
            best_diff = diff
            best_vec = vec
            best_alpha = mid
            best_energy = energy
        if n_sel < k:
            lo = mid
        else:
            hi = mid
        if diff == 0:
            break
    bar.close()
    print(f"  kept best_alpha={best_alpha:.6f} (|n-k|={best_diff}, E={best_energy:.4f})")
    Q = build_qubo_matrix(I, R, alpha=best_alpha, k=k)
    vec, energy = solve_qubo(
        Q, num_reads=num_reads, seed=seed, solver=solver, timeout_ms=timeout_ms
    )
    if abs(int(vec.sum()) - k) < abs(int(best_vec.sum()) - k) or (
        abs(int(vec.sum()) - k) == abs(int(best_vec.sum()) - k)
        and energy <= _energy(Q, best_vec)
    ):
        chosen, chosen_e = vec, energy
    else:
        chosen, chosen_e = best_vec, _energy(Q, best_vec)
    report = diagnose_qubo_solution(Q, chosen, energy=chosen_e, k=k)
    return chosen, chosen_e, best_alpha, Q, report
