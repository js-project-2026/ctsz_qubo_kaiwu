"""L2 logistic model for DPP-IV inhibitory peptides.

A peptide is a variable-length string over the 20 standard amino acids.
StackDPPIV turns that string into fifty model probabilities and stacks
them. This script fits one convex model on a fixed map phi(s):

- amino-acid composition (20)
- one-hot of the first residue (20)
- one-hot of the second residue (20); zeros when the peptide is shorter
- one-hot of the last residue (20)
- dipeptide composition (400), the paper's DPC
- log(length)

``dppiv_suite.py`` compares these blocks under L2 logistic, an RBF SVM,
and histogram gradient boosting, then refits after length-balanced
downsampling. Positives in the public split are much shorter than
negatives, so length is reported on its own.

The training problem is

    min_{w,b}  mean_i log(1 + exp(-(2 y_i - 1) (w · x_i + b)))
               + (lambda / 2) ||w||^2

with x_i the training-set z-score of phi(s_i). L-BFGS solves it. lambda
is chosen by stratified 5-fold MCC on the training set only. The
independent test set is scored once, at probability 0.5.

Run with ``--download`` to fetch the authors' own split, or pass the
four FASTA paths explicitly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA)}
N_AA = len(AA)
SEQ_DIM = 4 * N_AA  # AAC, N1, N2, C
DPC_DIM = N_AA * N_AA
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
# Bins used to hold length fixed. Positives cluster at 2–5, negatives at >= 11.
LENGTH_BINS = ((2, 5, "2-5"), (6, 10, "6-10"), (11, 10**9, "11+"))

DATA_BASE = "http://pmlabstack.pythonanywhere.com/static"
DATA_FILES = {
    "train_pos": "TR_pos_dppiv.txt",
    "train_neg": "TR_neg_dppiv.txt",
    "test_pos": "TS_pos_dppiv.txt",
    "test_neg": "TS_neg_dppiv.txt",
}
# Counts stated in the paper: 80:20 split of 665 positives and 665 negatives.
EXPECTED_COUNTS = {"train_pos": 532, "train_neg": 532, "test_pos": 133, "test_neg": 133}


def download_dataset(folder: Path) -> dict[str, Path]:
    """Fetch the four FASTA files published with StackDPPIV into ``folder``."""
    import urllib.request

    folder.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, name in DATA_FILES.items():
        target = folder / name
        if not target.is_file():
            url = f"{DATA_BASE}/{name}"
            print(f"Downloading {url}", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response:
                target.write_bytes(response.read())
        n_seqs = len(parse_fasta(target))
        if n_seqs != EXPECTED_COUNTS[key]:
            raise RuntimeError(
                f"{name} has {n_seqs} sequences, expected {EXPECTED_COUNTS[key]}"
            )
        paths[key] = target
    return paths


def parse_fasta(path: Path) -> list[str]:
    """Read a FASTA file of peptide sequences. Headers are discarded."""
    seqs: list[str] = []
    current: list[str] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                seqs.append("".join(current))
                current = []
            continue
        unknown = [ch for ch in line if ch not in AA_INDEX]
        if unknown:
            raise ValueError(f"Nonstandard residue {unknown[0]!r} in {path}")
        current.append(line)
    if current:
        seqs.append("".join(current))
    return seqs


def sequence_features(seqs: list[str]) -> np.ndarray:
    """Return (n, 80) features: composition and the three terminal residues."""
    out = np.zeros((len(seqs), SEQ_DIM), dtype=np.float64)
    for row, seq in enumerate(seqs):
        length = len(seq)
        if length == 0:
            raise ValueError("Empty peptide")
        for ch in seq:
            out[row, AA_INDEX[ch]] += 1.0
        out[row, :N_AA] /= length
        out[row, N_AA + AA_INDEX[seq[0]]] = 1.0
        if length >= 2:
            out[row, 2 * N_AA + AA_INDEX[seq[1]]] = 1.0
        out[row, 3 * N_AA + AA_INDEX[seq[-1]]] = 1.0
    return out


def dipeptide_features(seqs: list[str]) -> np.ndarray:
    """Return (n, 400) dipeptide composition. Each row sums to 1 when length >= 2."""
    out = np.zeros((len(seqs), N_AA * N_AA), dtype=np.float64)
    for row, seq in enumerate(seqs):
        pairs = len(seq) - 1
        if pairs <= 0:
            continue
        for left, right in zip(seq, seq[1:]):
            out[row, AA_INDEX[left] * N_AA + AA_INDEX[right]] += 1.0
        out[row] /= pairs
    return out


def _log_length(seqs: list[str]) -> np.ndarray:
    return np.log(np.array([len(seq) for seq in seqs], dtype=np.float64))[:, None]


def feature_matrix(seqs: list[str], mode: str) -> np.ndarray:
    """Build a design matrix.

    ``mode`` is ``len``, ``seq``, ``dpc``, or a ``+`` combination such as
    ``seq+dpc+len``. ``seq`` is composition plus the three terminal residues.
    ``dpc`` is the 400-dimensional dipeptide composition.
    """
    blocks = {
        "seq": sequence_features,
        "dpc": dipeptide_features,
        "len": _log_length,
    }
    names = [part for part in mode.split("+") if part]
    unknown = [name for name in names if name not in blocks]
    if unknown or not names:
        raise ValueError(f"Unknown feature mode {mode!r}")
    parts = [blocks[name](seqs) for name in names]
    if len(parts) == 1:
        return parts[0]
    return np.hstack(parts)


def length_bin_name(length: int) -> str:
    """Name of the length bin that contains ``length``."""
    for low, high, name in LENGTH_BINS:
        if low <= length <= high:
            return name
    raise ValueError(f"No length bin for {length}")


def balance_indices(lengths: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Downsample so each length bin contributes equally many positives and negatives.

    A bin that contains only one class is left out. The returned indices are sorted.
    """
    lengths = np.asarray(lengths)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for low, high, _name in LENGTH_BINS:
        in_bin = np.flatnonzero((lengths >= low) & (lengths <= high))
        positive = in_bin[y[in_bin] == 1]
        negative = in_bin[y[in_bin] == 0]
        n_keep = min(positive.size, negative.size)
        if n_keep == 0:
            continue
        chosen.append(rng.choice(positive, n_keep, replace=False))
        chosen.append(rng.choice(negative, n_keep, replace=False))
    if not chosen:
        raise RuntimeError("No length bin contains both classes")
    return np.sort(np.concatenate(chosen))


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return mu, sigma


def _standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _objective(theta: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float):
    w = theta[:-1]
    b = theta[-1]
    z = X @ w + b
    # d/dz log(1+exp(-(2y-1) z)) = sigmoid(z) - y
    residual = _sigmoid(z) - y
    loss = np.mean(np.maximum(z, 0) - y * z + np.log1p(np.exp(-np.abs(z))))
    # The identity above is the logistic loss for labels in {0, 1}:
    # softplus(z) - y z. Equivalent to log(1+exp(-(2y-1) z)).
    reg = 0.5 * lam * float(w @ w)
    grad = np.empty_like(theta)
    grad[:-1] = (X.T @ residual) / y.size + lam * w
    grad[-1] = float(residual.mean())
    return loss + reg, grad


def fit_logistic(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Return theta = (w, b) for standardized columns of X."""
    theta0 = np.zeros(X.shape[1] + 1, dtype=np.float64)
    result = minimize(
        _objective,
        theta0,
        args=(X, y, lam),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 400},
    )
    if not result.success:
        raise RuntimeError(f"L-BFGS failed at lambda={lam}: {result.message}")
    return result.x


def predict_proba(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return _sigmoid(X @ theta[:-1] + theta[-1])


def binary_metrics(y: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """ACC, sensitivity, specificity, MCC, and AUC at a fixed probability cut."""
    y = np.asarray(y, dtype=np.int64)
    pred = (proba >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    total = tp + tn + fp + fn
    acc = (tp + tn) / total
    sn = tp / (tp + fn) if (tp + fn) else float("nan")
    sp = tn / (tn + fp) if (tn + fp) else float("nan")
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / np.sqrt(denom) if denom else float("nan")
    return {
        "ACC": acc,
        "Sn": sn,
        "Sp": sp,
        "MCC": mcc,
        "AUC": roc_auc(y, proba),
    }


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """AUC by the Mann-Whitney statistic. Ties count as one half."""
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    pos = score[y == 1]
    neg = score[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, dtype=np.float64)
    sorted_score = score[order]
    start = 0
    while start < score.size:
        stop = start + 1
        while stop < score.size and sorted_score[stop] == sorted_score[start]:
            stop += 1
        # Average rank, 1-based, for the tied block.
        avg = 0.5 * (start + 1 + stop)
        ranks[order[start:stop]] = avg
        start = stop
    sum_pos = float(ranks[y == 1].sum())
    return (sum_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)


def stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        for i, row in enumerate(idx):
            folds[i % n_folds].append(int(row))
    return [np.array(fold, dtype=np.int64) for fold in folds]


def select_lambda(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: tuple[float, ...] = LAMBDAS,
    n_folds: int = 5,
    seed: int = 0,
) -> tuple[float, float]:
    """Pick lambda by mean validation MCC. Returns (lambda, mean MCC)."""
    folds = stratified_folds(y, n_folds, seed)
    best_lam = lambdas[0]
    best_mcc = -np.inf
    for lam in lambdas:
        scores = []
        for hold in range(n_folds):
            val = folds[hold]
            train = np.concatenate([folds[i] for i in range(n_folds) if i != hold])
            mu, sigma = _standardize_fit(X[train])
            theta = fit_logistic(_standardize_apply(X[train], mu, sigma), y[train], lam)
            proba = predict_proba(_standardize_apply(X[val], mu, sigma), theta)
            scores.append(binary_metrics(y[val], proba)["MCC"])
        mean_mcc = float(np.mean(scores))
        if mean_mcc > best_mcc:
            best_mcc = mean_mcc
            best_lam = lam
    return best_lam, best_mcc


def fit_predict(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit on the training matrix and return (train proba, test proba, theta)."""
    mu, sigma = _standardize_fit(train_X)
    theta = fit_logistic(_standardize_apply(train_X, mu, sigma), train_y, lam)
    train_p = predict_proba(_standardize_apply(train_X, mu, sigma), theta)
    test_p = predict_proba(_standardize_apply(test_X, mu, sigma), theta)
    return train_p, test_p, theta


def _format_metrics(title: str, metrics: dict[str, float]) -> str:
    return (
        f"{title}: ACC={metrics['ACC']:.3f}  Sn={metrics['Sn']:.3f}  "
        f"Sp={metrics['Sp']:.3f}  MCC={metrics['MCC']:.3f}  AUC={metrics['AUC']:.3f}"
    )


def run(train_pos, train_neg, test_pos, test_neg) -> None:
    train_seqs = parse_fasta(train_pos) + parse_fasta(train_neg)
    test_seqs = parse_fasta(test_pos) + parse_fasta(test_neg)
    n_pos_tr = len(parse_fasta(train_pos))
    n_pos_te = len(parse_fasta(test_pos))
    train_y = np.array([1] * n_pos_tr + [0] * (len(train_seqs) - n_pos_tr))
    test_y = np.array([1] * n_pos_te + [0] * (len(test_seqs) - n_pos_te))
    print(
        f"Train {n_pos_tr}+{len(train_seqs) - n_pos_tr}  "
        f"test {n_pos_te}+{len(test_seqs) - n_pos_te}"
    )
    print(
        "StackDPPIV independent test (reported): "
        "ACC=0.891  Sn=0.857  Sp=0.925  MCC=0.784  AUC=0.961"
    )
    for mode in ("len", "seq", "seq+len"):
        train_X = feature_matrix(train_seqs, mode)
        test_X = feature_matrix(test_seqs, mode)
        lam, cv_mcc = select_lambda(train_X, train_y)
        train_p, test_p, _ = fit_predict(train_X, train_y, test_X, lam)
        print(f"mode={mode}  lambda={lam:g}  CV_MCC={cv_mcc:.3f}")
        print("  " + _format_metrics("train", binary_metrics(train_y, train_p)))
        print("  " + _format_metrics("test", binary_metrics(test_y, test_p)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch the four published FASTA files instead of passing paths",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/dppiv"),
        help="Where --download caches the FASTA files",
    )
    parser.add_argument("--train-pos", type=Path)
    parser.add_argument("--train-neg", type=Path)
    parser.add_argument("--test-pos", type=Path)
    parser.add_argument("--test-neg", type=Path)
    args = parser.parse_args()
    if args.download:
        paths = download_dataset(args.data_dir)
        run(
            paths["train_pos"],
            paths["train_neg"],
            paths["test_pos"],
            paths["test_neg"],
        )
        return
    missing = [
        name
        for name, value in (
            ("--train-pos", args.train_pos),
            ("--train-neg", args.train_neg),
            ("--test-pos", args.test_pos),
            ("--test-neg", args.test_neg),
        )
        if value is None
    ]
    if missing:
        parser.error(f"Pass --download or all of: {', '.join(missing)}")
    run(args.train_pos, args.train_neg, args.test_pos, args.test_neg)


if __name__ == "__main__":
    main()
