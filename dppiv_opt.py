"""Train-only threshold, a small stack, and the descriptors the paper adds.

Every choice below is fixed before the independent test is scored, or is
chosen from training out-of-fold MCC. The test set is read only in the
final report.

    python dppiv_opt.py --download

Stages, in the order fixed before this run:

1. Histogram gradient boosting on seq+dpc. Depth is 3 or 6, and the
   probability cut is chosen by out-of-fold MCC.
2. A meta-model on boosting, extra trees and an RBF SVM, each fit to
   seq, dpc and seq+dpc.
3. The same meta-model with CTD, PAAC, APAAC and the 11 amino-acid scales.
4. Chemical fingerprints join the pool: EState, MACCS, PubChem, FP4.
5. The same pool, after keeping the top 10, 15, 20 or all base models,
   whichever training MCC prefers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dppiv_features import (
    AAI_DIM,
    APAAC_DIM,
    CTD_DIM,
    PAAC_DIM,
    aai_features,
    apaac_features,
    ctd_features,
    fingerprint_features,
    paac_features,
)
from dppiv_logit import (
    binary_metrics,
    download_dataset,
    feature_matrix,
    fit_predict,
    parse_fasta,
    select_threshold,
    stratified_folds,
)

# Locked from the earlier train-only grid: depth 6 won for seq, dpc and seq+dpc.
HGB_DEPTH = 6
HGB_KW = dict(learning_rate=0.08, max_iter=200, min_samples_leaf=15, random_state=0)
RBF_C = 1.0
RBF_MAX_DIM = 500
META_LAMBDAS = (0.01, 0.1, 1.0)
SUBSET_SIZES = (10, 15, 20)


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


def _hgb_proba(train_X, train_y, eval_X, depth: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(max_depth=depth, **HGB_KW)
    model.fit(train_X, train_y)
    return model.predict_proba(eval_X)[:, 1]


def _et_proba(train_X, train_y, eval_X):
    from sklearn.ensemble import ExtraTreesClassifier

    model = ExtraTreesClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=0,
        n_jobs=4,
    )
    model.fit(train_X, train_y)
    return model.predict_proba(eval_X)[:, 1]


def _rbf_proba(train_X, train_y, eval_X):
    from sklearn.svm import SVC

    from dppiv_logit import _sigmoid, _standardize_apply, _standardize_fit

    mu, sigma = _standardize_fit(train_X)
    model = SVC(kernel="rbf", C=RBF_C, gamma="scale", max_iter=4000)
    model.fit(_standardize_apply(train_X, mu, sigma), train_y)
    decision = model.decision_function(_standardize_apply(eval_X, mu, sigma))
    return _sigmoid(np.asarray(decision, dtype=np.float64))


def _linear_proba(train_X, train_y, eval_X):
    from sklearn.linear_model import LogisticRegression

    from dppiv_logit import _standardize_apply, _standardize_fit

    mu, sigma = _standardize_fit(train_X)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
    model.fit(_standardize_apply(train_X, mu, sigma), train_y)
    return model.predict_proba(_standardize_apply(eval_X, mu, sigma))[:, 1]


def _predict_oof(X, y, folds, trainer) -> np.ndarray:
    proba = np.zeros(len(y), dtype=np.float64)
    for hold in range(len(folds)):
        val = folds[hold]
        train = np.concatenate([folds[i] for i in range(len(folds)) if i != hold])
        proba[val] = trainer(X[train], y[train], X[val])
    return proba


def _meta_hgb(train_X, train_y, eval_X):
    return _hgb_proba(train_X, train_y, eval_X, depth=2)


def _choose_meta(oof: np.ndarray, y: np.ndarray, folds):
    """Return (name, threshold, train MCC, predictor of eval rows)."""
    best = None
    for lam in META_LAMBDAS:
        pred = _predict_oof(
            oof,
            y,
            folds,
            lambda Xtr, ytr, Xva, lam=lam: fit_predict(Xtr, ytr, Xva, lam)[1],
        )
        threshold, mcc = select_threshold(y, pred)
        if best is None or mcc > best[2]:
            best = (f"logistic:{lam:g}", threshold, mcc, "logistic", lam)
    pred = _predict_oof(oof, y, folds, _meta_hgb)
    threshold, mcc = select_threshold(y, pred)
    if best is None or mcc > best[2]:
        best = ("hgb:2", threshold, mcc, "hgb", None)
    return best


def _apply_meta(kind: str, lam, oof, y, eval_X):
    if kind == "logistic":
        return fit_predict(oof, y, eval_X, lam)[1]
    return _meta_hgb(oof, y, eval_X)


def _score_block(name, columns, oof, test_p, y, test_y, folds) -> str:
    chosen = _choose_meta(oof[:, columns], y, folds)
    label, threshold, train_mcc, kind, lam = chosen
    held = _apply_meta(kind, lam, oof[:, columns], y, test_p[:, columns])
    metrics = binary_metrics(test_y, held, threshold=threshold)
    text = (
        f"{name}: meta={label}  threshold={threshold:.2f}  "
        f"train_MCC={train_mcc:.3f}  n_base={len(columns)}  " + _line("test", metrics)
    )
    print(text, flush=True)
    return text


def _build_matrices(train_seqs, test_seqs):
    """Feature blocks used by the base models. Computed once."""
    blocks = {
        "seq": (
            feature_matrix(train_seqs, "seq"),
            feature_matrix(test_seqs, "seq"),
        ),
        "dpc": (
            feature_matrix(train_seqs, "dpc"),
            feature_matrix(test_seqs, "dpc"),
        ),
        "seq+dpc": (
            feature_matrix(train_seqs, "seq+dpc"),
            feature_matrix(test_seqs, "seq+dpc"),
        ),
    }
    print("encoding CTD / PAAC / APAAC / AAI", flush=True)
    blocks["ctd"] = (ctd_features(train_seqs), ctd_features(test_seqs))
    blocks["paac"] = (paac_features(train_seqs), paac_features(test_seqs))
    blocks["apaac"] = (apaac_features(train_seqs), apaac_features(test_seqs))
    blocks["aai"] = (aai_features(train_seqs), aai_features(test_seqs))
    for kind in ("estate", "maccs", "pubchem", "fp4"):
        print(f"encoding {kind}", flush=True)
        both = train_seqs + test_seqs
        encoded = fingerprint_features(both, kind)
        blocks[kind] = (encoded[: len(train_seqs)], encoded[len(train_seqs) :])
    expected = {
        "ctd": CTD_DIM,
        "paac": PAAC_DIM,
        "apaac": APAAC_DIM,
        "aai": AAI_DIM,
        "estate": 79,
        "maccs": 166,
        "pubchem": 881,
        "fp4": 307,
    }
    for name, width in expected.items():
        if blocks[name][0].shape[1] != width:
            raise RuntimeError(f"{name} has width {blocks[name][0].shape[1]}, expected {width}")
    return blocks


def _algo_for(feature: str, width: int) -> tuple[str, ...]:
    if width > RBF_MAX_DIM:
        return ("hgb", "et", "linear")
    return ("hgb", "et", "rbf")


def _trainer(algo: str):
    if algo == "hgb":
        return lambda Xtr, ytr, Xva: _hgb_proba(Xtr, ytr, Xva, HGB_DEPTH)
    if algo == "et":
        return _et_proba
    if algo == "rbf":
        return _rbf_proba
    if algo == "linear":
        return _linear_proba
    raise ValueError(algo)


def run_opt(train_pos, train_neg, test_pos, test_neg) -> None:
    train_seqs, train_y, test_seqs, test_y = load_split(
        train_pos, train_neg, test_pos, test_neg
    )
    folds = stratified_folds(train_y, 5, seed=0)
    print(
        f"Train {int(train_y.sum())}+{int((train_y == 0).sum())}  "
        f"test {int(test_y.sum())}+{int((test_y == 0).sum())}",
        flush=True,
    )
    print(
        "StackDPPIV Table 3: ACC=0.891  Sn=0.857  Sp=0.925  MCC=0.784  AUC=0.961",
        flush=True,
    )
    blocks = _build_matrices(train_seqs, test_seqs)

    print("\n=== stage 1: HGB seq+dpc, threshold from training OOF ===", flush=True)
    train_X, test_X = blocks["seq+dpc"]
    stage1 = None
    for depth in (3, 6):
        oof = _predict_oof(
            train_X,
            train_y,
            folds,
            lambda Xtr, ytr, Xva, depth=depth: _hgb_proba(Xtr, ytr, Xva, depth),
        )
        threshold, train_mcc = select_threshold(train_y, oof)
        print(
            f"  depth={depth}  threshold={threshold:.2f}  train_MCC={train_mcc:.3f}",
            flush=True,
        )
        if stage1 is None or train_mcc > stage1[0]:
            stage1 = (train_mcc, depth, threshold)
    _, depth, threshold = stage1
    test_p = _hgb_proba(train_X, train_y, test_X, depth)
    tuned = binary_metrics(test_y, test_p, threshold=threshold)
    at_half = binary_metrics(test_y, test_p, threshold=0.5)
    print(
        f"stage1 chosen depth={depth}  threshold={threshold:.2f}  " + _line("test", tuned),
        flush=True,
    )
    print("stage1 same scores at 0.50  " + _line("test", at_half), flush=True)

    print("\n=== base models, training OOF only ===", flush=True)
    names: list[str] = []
    oof_cols = []
    test_cols = []
    groups = {
        "core": [],
        "desc": [],
        "chem": [],
    }
    group_of = {
        "seq": "core",
        "dpc": "core",
        "seq+dpc": "core",
        "ctd": "desc",
        "paac": "desc",
        "apaac": "desc",
        "aai": "desc",
        "estate": "chem",
        "maccs": "chem",
        "pubchem": "chem",
        "fp4": "chem",
    }
    for feature, (train_block, test_block) in blocks.items():
        for algo in _algo_for(feature, train_block.shape[1]):
            name = f"{algo}:{feature}"
            print(f"  fitting {name}  dim={train_block.shape[1]}", flush=True)
            trainer = _trainer(algo)
            oof = _predict_oof(train_block, train_y, folds, trainer)
            held = trainer(train_block, train_y, test_block)
            mcc = binary_metrics(train_y, oof)["MCC"]
            print(f"    OOF_MCC={mcc:.3f}", flush=True)
            groups[group_of[feature]].append(len(names))
            names.append(name)
            oof_cols.append(oof)
            test_cols.append(held)
    oof = np.column_stack(oof_cols)
    test_p = np.column_stack(test_cols)

    print("\n=== stages 2–5, each test score is a single look ===", flush=True)
    core = groups["core"]
    desc = groups["desc"]
    chem = groups["chem"]
    _score_block("stage2 core stack", core, oof, test_p, train_y, test_y, folds)
    _score_block(
        "stage3 core+descriptors",
        core + desc,
        oof,
        test_p,
        train_y,
        test_y,
        folds,
    )
    all_cols = core + desc + chem
    _score_block("stage4 full stack", all_cols, oof, test_p, train_y, test_y, folds)

    ranking = sorted(
        all_cols,
        key=lambda j: binary_metrics(train_y, oof[:, j])["MCC"],
        reverse=True,
    )
    best_subset = None
    for size in list(SUBSET_SIZES) + [len(ranking)]:
        chosen_cols = ranking[:size]
        picked = _choose_meta(oof[:, chosen_cols], train_y, folds)
        print(
            f"  subset m={size}  meta={picked[0]}  threshold={picked[1]:.2f}  "
            f"train_MCC={picked[2]:.3f}",
            flush=True,
        )
        if best_subset is None or picked[2] > best_subset[3]:
            best_subset = (size, picked[0], picked[1], picked[2], chosen_cols, picked)
    size, _label, _threshold, _mcc, chosen_cols, picked = best_subset
    kept = ", ".join(names[j] for j in chosen_cols)
    print(f"stage5 keeps m={size}: {kept}", flush=True)
    label, threshold, train_mcc, kind, lam = picked
    held = _apply_meta(kind, lam, oof[:, chosen_cols], train_y, test_p[:, chosen_cols])
    metrics = binary_metrics(test_y, held, threshold=threshold)
    print(
        f"stage5 chosen m={size}  meta={label}  threshold={threshold:.2f}  "
        f"train_MCC={train_mcc:.3f}  " + _line("test", metrics),
        flush=True,
    )


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
        run_opt(paths["train_pos"], paths["train_neg"], paths["test_pos"], paths["test_neg"])
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
    run_opt(args.train_pos, args.train_neg, args.test_pos, args.test_neg)


if __name__ == "__main__":
    main()
