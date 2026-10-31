"""Dipeptide features, RBF SVM, gradient boosting, and a length-balanced check.

StackDPPIV's independent-test numbers stay the reference. Every
hyperparameter is chosen by stratified 5-fold MCC on the training split.
The independent test is scored once.

    python dppiv_suite.py --download
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dppiv_logit import (
    LENGTH_BINS,
    balance_indices,
    binary_metrics,
    download_dataset,
    feature_matrix,
    fit_predict,
    parse_fasta,
    select_lambda,
    stratified_folds,
    _sigmoid,
    _standardize_apply,
    _standardize_fit,
)

LOGISTIC_MODES = (
    "len",
    "seq",
    "dpc",
    "seq+dpc",
    "seq+len",
    "dpc+len",
    "seq+dpc+len",
)
NONLINEAR_MODES = ("seq", "dpc", "seq+dpc", "seq+len", "dpc+len", "seq+dpc+len")
STACK_LINE = (
    "StackDPPIV reported: ACC=0.891  Sn=0.857  Sp=0.925  MCC=0.784  AUC=0.961"
)


def load_split(train_pos, train_neg, test_pos, test_neg):
    train_pos_seq = parse_fasta(train_pos)
    train_neg_seq = parse_fasta(train_neg)
    test_pos_seq = parse_fasta(test_pos)
    test_neg_seq = parse_fasta(test_neg)
    train_seqs = train_pos_seq + train_neg_seq
    test_seqs = test_pos_seq + test_neg_seq
    train_y = np.array([1] * len(train_pos_seq) + [0] * len(train_neg_seq))
    test_y = np.array([1] * len(test_pos_seq) + [0] * len(test_neg_seq))
    return train_seqs, train_y, test_seqs, test_y


def _line(title: str, metrics: dict[str, float]) -> str:
    return (
        f"{title}: ACC={metrics['ACC']:.3f}  Sn={metrics['Sn']:.3f}  "
        f"Sp={metrics['Sp']:.3f}  MCC={metrics['MCC']:.3f}  AUC={metrics['AUC']:.3f}"
    )


def _mean_cv_mcc(fold_proba, X, y) -> float:
    folds = stratified_folds(y, 5, seed=0)
    scores = []
    for hold in range(5):
        val = folds[hold]
        train = np.concatenate([folds[i] for i in range(5) if i != hold])
        proba = fold_proba(X[train], y[train], X[val])
        scores.append(binary_metrics(y[val], proba)["MCC"])
    return float(np.mean(scores))


def logistic_test(mode, train_seqs, train_y, test_seqs, test_y, label="logistic"):
    train_X = feature_matrix(train_seqs, mode)
    test_X = feature_matrix(test_seqs, mode)
    lam, cv_mcc = select_lambda(train_X, train_y)
    _train_p, test_p, _theta = fit_predict(train_X, train_y, test_X, lam)
    metrics = binary_metrics(test_y, test_p)
    print(
        f"{label} {mode:12s}  lambda={lam:g}  CV_MCC={cv_mcc:.3f}  "
        + _line("test", metrics),
        flush=True,
    )
    return test_p, metrics


def _svm_proba(train_X, train_y, eval_X, C: float):
    from sklearn.svm import SVC

    mu, sigma = _standardize_fit(train_X)
    model = SVC(kernel="rbf", C=C, gamma="scale")
    model.fit(_standardize_apply(train_X, mu, sigma), train_y)
    decision = model.decision_function(_standardize_apply(eval_X, mu, sigma))
    return _sigmoid(np.asarray(decision, dtype=np.float64))


def svm_test(mode, train_seqs, train_y, test_seqs, test_y):
    train_X = feature_matrix(train_seqs, mode)
    test_X = feature_matrix(test_seqs, mode)
    best_C = 0.1
    best_mcc = -np.inf
    for C in (0.1, 1.0, 10.0):
        score = _mean_cv_mcc(
            lambda Xtr, ytr, Xva, C=C: _svm_proba(Xtr, ytr, Xva, C),
            train_X,
            train_y,
        )
        if score > best_mcc:
            best_mcc = score
            best_C = C
    test_p = _svm_proba(train_X, train_y, test_X, best_C)
    metrics = binary_metrics(test_y, test_p)
    print(
        f"rbf-svm  {mode:12s}  C={best_C:g}  CV_MCC={best_mcc:.3f}  "
        + _line("test", metrics),
        flush=True,
    )
    return test_p, metrics


def _hgb_proba(train_X, train_y, eval_X, depth: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_depth=depth,
        learning_rate=0.08,
        max_iter=250,
        min_samples_leaf=15,
        random_state=0,
    )
    model.fit(train_X, train_y)
    return model.predict_proba(eval_X)[:, 1]


def hgb_test(mode, train_seqs, train_y, test_seqs, test_y):
    train_X = feature_matrix(train_seqs, mode)
    test_X = feature_matrix(test_seqs, mode)
    best_depth = 3
    best_mcc = -np.inf
    for depth in (3, 6):
        score = _mean_cv_mcc(
            lambda Xtr, ytr, Xva, depth=depth: _hgb_proba(Xtr, ytr, Xva, depth),
            train_X,
            train_y,
        )
        if score > best_mcc:
            best_mcc = score
            best_depth = depth
    test_p = _hgb_proba(train_X, train_y, test_X, best_depth)
    metrics = binary_metrics(test_y, test_p)
    print(
        f"hgb      {mode:12s}  depth={best_depth}  CV_MCC={best_mcc:.3f}  "
        + _line("test", metrics),
        flush=True,
    )
    return test_p, metrics


def print_length_counts(title: str, seqs: list[str], y: np.ndarray) -> None:
    lengths = np.array([len(seq) for seq in seqs])
    parts = [title]
    for low, high, name in LENGTH_BINS:
        mask = (lengths >= low) & (lengths <= high)
        parts.append(
            f"{name} pos={int((mask & (y == 1)).sum())} neg={int((mask & (y == 0)).sum())}"
        )
    print("  ".join(parts), flush=True)


def print_within_bins(title: str, seqs: list[str], y: np.ndarray, proba: np.ndarray) -> None:
    lengths = np.array([len(seq) for seq in seqs])
    print(title, flush=True)
    for low, high, name in LENGTH_BINS:
        mask = (lengths >= low) & (lengths <= high)
        n_pos = int((mask & (y == 1)).sum())
        n_neg = int((mask & (y == 0)).sum())
        if n_pos == 0 or n_neg == 0:
            print(f"  {name}: pos={n_pos} neg={n_neg}  skipped", flush=True)
            continue
        metrics = binary_metrics(y[mask], proba[mask])
        print(
            f"  {name} pos={n_pos} neg={n_neg}  " + _line("test", metrics),
            flush=True,
        )


def balanced_logistic(mode, train_seqs, train_y, test_seqs, test_y):
    lengths = np.array([len(seq) for seq in train_seqs])
    keep = balance_indices(lengths, train_y, seed=0)
    kept_pos = int((train_y[keep] == 1).sum())
    print(
        f"length-balanced train n={keep.size} pos={kept_pos} neg={keep.size - kept_pos}",
        flush=True,
    )
    train_kept = [train_seqs[i] for i in keep]
    y_kept = train_y[keep]
    return logistic_test(
        mode, train_kept, y_kept, test_seqs, test_y, label="balanced"
    )


def run_suite(train_pos, train_neg, test_pos, test_neg) -> None:
    train_seqs, train_y, test_seqs, test_y = load_split(
        train_pos, train_neg, test_pos, test_neg
    )
    print(
        f"Train {int(train_y.sum())}+{int((train_y == 0).sum())}  "
        f"test {int(test_y.sum())}+{int((test_y == 0).sum())}",
        flush=True,
    )
    print(STACK_LINE, flush=True)
    print_length_counts("train lengths", train_seqs, train_y)
    print_length_counts("test lengths", test_seqs, test_y)

    print("\n=== L2 logistic ===", flush=True)
    logistic_scores = {}
    for mode in LOGISTIC_MODES:
        proba, _metrics = logistic_test(mode, train_seqs, train_y, test_seqs, test_y)
        logistic_scores[mode] = proba

    print("\n=== RBF SVM ===", flush=True)
    for mode in NONLINEAR_MODES:
        svm_test(mode, train_seqs, train_y, test_seqs, test_y)

    print("\n=== histogram gradient boosting ===", flush=True)
    for mode in NONLINEAR_MODES:
        hgb_test(mode, train_seqs, train_y, test_seqs, test_y)

    print("\n=== within-bin test, logistic trained on the full split ===", flush=True)
    for mode in ("seq", "dpc", "seq+dpc"):
        print_within_bins(
            f"logistic {mode}",
            test_seqs,
            test_y,
            logistic_scores[mode],
        )

    print("\n=== logistic refit after length-balanced downsampling ===", flush=True)
    for mode in ("seq", "dpc", "seq+dpc"):
        proba, _metrics = balanced_logistic(
            mode, train_seqs, train_y, test_seqs, test_y
        )
        print_within_bins(f"  within-bin {mode}", test_seqs, test_y, proba)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("data/dppiv"))
    parser.add_argument("--train-pos", type=Path)
    parser.add_argument("--train-neg", type=Path)
    parser.add_argument("--test-pos", type=Path)
    parser.add_argument("--test-neg", type=Path)
    args = parser.parse_args()
    if args.download:
        paths = download_dataset(args.data_dir)
        run_suite(
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
    run_suite(args.train_pos, args.train_neg, args.test_pos, args.test_neg)


if __name__ == "__main__":
    main()
