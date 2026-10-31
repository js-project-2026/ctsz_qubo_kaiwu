"""Four strong boosters, an N-terminal dipeptide, then the paper's 50 models.

Threshold stays at 0.5. Out-of-fold predictions use 10 stratified folds.
The independent test is scored only after the training rule has decided
whether the 50-model stack is required.

    python dppiv_narrow.py --download

The training bar is 0.78. Steps 1 and 2 are a four-model stack and that
stack plus the first dipeptide. If the better training MCC is still below
0.78, step 3 fits the paper's ten encodings with extra trees, logistic
regression, a multilayer perceptron, a random forest and an SVM, and an
inner fold chooses how many of those probabilities to keep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dppiv_features import (
    aai_features,
    apaac_features,
    ctd_features,
    fingerprint_features,
    nterm_dipeptide_features,
    paac_features,
)
from dppiv_logit import (
    N_AA,
    binary_metrics,
    download_dataset,
    feature_matrix,
    fit_predict,
    stratified_folds,
)
from dppiv_opt import (
    HGB_DEPTH,
    _et_proba,
    _hgb_proba,
    _linear_proba,
    _line,
    _predict_oof,
    _rbf_proba,
    load_split,
)

TRAIN_BAR = 0.78
META_LAMBDAS = (0.1, 1.0)
SUBSET_SIZES = (10, 15, 20, 30)
CORE = ("hgb:seq", "hgb:seq+dpc", "hgb:ctd", "hgb:estate")
PAPER_FEATURES = (
    "aac",
    "dpc",
    "aai",
    "apaac",
    "ctd",
    "paac",
    "estate",
    "fp4",
    "maccs",
    "pubchem",
)
PAPER_ALGOS = ("et", "lr", "mlp", "rf", "svm")


def _mcc(y, proba) -> float:
    score = binary_metrics(y, proba)["MCC"]
    if not np.isfinite(score):
        return -1.0
    return float(score)


def _concat(folds, hold: int) -> np.ndarray:
    return np.concatenate([folds[i] for i in range(len(folds)) if i != hold])


def _meta_inner_mcc(oof: np.ndarray, y: np.ndarray, lam: float, folds) -> float:
    pred = np.zeros(len(y), dtype=np.float64)
    for hold in range(len(folds)):
        val = folds[hold]
        train = _concat(folds, hold)
        pred[val] = fit_predict(oof[train], y[train], oof[val], lam)[1]
    return _mcc(y, pred)


def _choose_lambda(oof: np.ndarray, y: np.ndarray, folds) -> tuple[float, float]:
    best_lam = META_LAMBDAS[0]
    best_mcc = -np.inf
    for lam in META_LAMBDAS:
        score = _meta_inner_mcc(oof, y, lam, folds)
        if score > best_mcc:
            best_mcc = score
            best_lam = lam
    return best_lam, best_mcc


def _rank_columns(oof: np.ndarray, y: np.ndarray) -> np.ndarray:
    scores = np.array([_mcc(y, oof[:, j]) for j in range(oof.shape[1])])
    return np.argsort(-scores)


def _select_subset(oof: np.ndarray, y: np.ndarray, folds) -> tuple[int, float, float]:
    """Pick how many probability columns to keep. Returns (k, lambda, inner MCC)."""
    sizes = [size for size in SUBSET_SIZES if size < oof.shape[1]]
    sizes.append(oof.shape[1])
    best = (oof.shape[1], META_LAMBDAS[0], -np.inf)
    for size in sizes:
        for lam in META_LAMBDAS:
            fold_scores = []
            for hold in range(len(folds)):
                val = folds[hold]
                train = _concat(folds, hold)
                chosen = _rank_columns(oof[train], y[train])[:size]
                pred = fit_predict(
                    oof[train][:, chosen], y[train], oof[val][:, chosen], lam
                )[1]
                fold_scores.append(_mcc(y[val], pred))
            score = float(np.mean(fold_scores))
            print(
                f"  inner k={size:<2}  lambda={lam:g}  MCC={score:.3f}",
                flush=True,
            )
            if score > best[2]:
                best = (size, lam, score)
    return best


def _rf_proba(train_X, train_y, eval_X):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=0,
        n_jobs=4,
    )
    model.fit(train_X, train_y)
    return model.predict_proba(eval_X)[:, 1]


def _mlp_proba(train_X, train_y, eval_X):
    from sklearn.neural_network import MLPClassifier

    from dppiv_logit import _standardize_apply, _standardize_fit

    mu, sigma = _standardize_fit(train_X)
    model = MLPClassifier(
        hidden_layer_sizes=(64,),
        alpha=1e-4,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=12,
        random_state=0,
    )
    model.fit(_standardize_apply(train_X, mu, sigma), train_y)
    return model.predict_proba(_standardize_apply(eval_X, mu, sigma))[:, 1]


def _svm_proba(train_X, train_y, eval_X):
    if train_X.shape[1] > 500:
        from sklearn.svm import LinearSVC

        from dppiv_logit import _sigmoid, _standardize_apply, _standardize_fit

        mu, sigma = _standardize_fit(train_X)
        model = LinearSVC(C=1.0, dual=False, max_iter=4000, random_state=0)
        model.fit(_standardize_apply(train_X, mu, sigma), train_y)
        decision = model.decision_function(_standardize_apply(eval_X, mu, sigma))
        return _sigmoid(np.asarray(decision, dtype=np.float64))
    return _rbf_proba(train_X, train_y, eval_X)


def _trainer(algo: str):
    if algo == "hgb":
        return lambda Xtr, ytr, Xva: _hgb_proba(Xtr, ytr, Xva, HGB_DEPTH)
    if algo == "et":
        return _et_proba
    if algo == "lr":
        return _linear_proba
    if algo == "mlp":
        return _mlp_proba
    if algo == "rf":
        return _rf_proba
    if algo == "svm":
        return _svm_proba
    raise ValueError(algo)


def _pair(train_seqs, test_seqs, encode):
    train_X = encode(train_seqs)
    test_X = encode(test_seqs)
    return train_X, test_X


def _fingerprint_pair(train_seqs, test_seqs, kind: str):
    encoded = fingerprint_features(train_seqs + test_seqs, kind)
    return encoded[: len(train_seqs)], encoded[len(train_seqs) :]


def _blocks(train_seqs, test_seqs) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    print("encoding sequence blocks", flush=True)
    seq_train = feature_matrix(train_seqs, "seq")
    seq_test = feature_matrix(test_seqs, "seq")
    dpc_train = feature_matrix(train_seqs, "dpc")
    dpc_test = feature_matrix(test_seqs, "dpc")
    nterm_train, nterm_test = _pair(train_seqs, test_seqs, nterm_dipeptide_features)
    blocks = {
        "seq": (seq_train, seq_test),
        "dpc": (dpc_train, dpc_test),
        "aac": (seq_train[:, :N_AA], seq_test[:, :N_AA]),
        "seq+dpc": (
            np.hstack([seq_train, dpc_train]),
            np.hstack([seq_test, dpc_test]),
        ),
        "nterm": (nterm_train, nterm_test),
        "seq+dpc+nterm": (
            np.hstack([seq_train, dpc_train, nterm_train]),
            np.hstack([seq_test, dpc_test, nterm_test]),
        ),
        "ctd": _pair(train_seqs, test_seqs, ctd_features),
        "paac": _pair(train_seqs, test_seqs, paac_features),
        "apaac": _pair(train_seqs, test_seqs, apaac_features),
        "aai": _pair(train_seqs, test_seqs, aai_features),
    }
    for kind in ("estate", "maccs", "pubchem", "fp4"):
        print(f"encoding {kind}", flush=True)
        blocks[kind] = _fingerprint_pair(train_seqs, test_seqs, kind)
    return blocks


class _Views:
    def __init__(self, blocks, y, folds):
        self.blocks = blocks
        self.y = y
        self.folds = folds
        self.oof: dict[str, np.ndarray] = {}
        self.test: dict[str, np.ndarray] = {}
        self._spec: dict[str, tuple[str, str]] = {}

    def ensure_oof(self, algo: str, feature: str) -> np.ndarray:
        name = f"{algo}:{feature}"
        if name in self.oof:
            return self.oof[name]
        train_X, _test_X = self.blocks[feature]
        print(f"  fitting {name}  dim={train_X.shape[1]}", flush=True)
        proba = _predict_oof(train_X, self.y, self.folds, _trainer(algo))
        self.oof[name] = proba
        self._spec[name] = (algo, feature)
        print(f"    OOF_MCC={_mcc(self.y, proba):.3f}", flush=True)
        return proba

    def ensure_test(self, name: str) -> np.ndarray:
        if name not in self.test:
            algo, feature = self._spec[name]
            train_X, test_X = self.blocks[feature]
            self.test[name] = _trainer(algo)(train_X, self.y, test_X)
        return self.test[name]


def _stack_matrix(views: _Views, names: tuple[str, ...] | list[str], which: str) -> np.ndarray:
    columns = []
    for name in names:
        algo, feature = name.split(":", 1)
        if which == "oof":
            columns.append(views.ensure_oof(algo, feature))
        else:
            views.ensure_oof(algo, feature)
            columns.append(views.ensure_test(name))
    return np.column_stack(columns)


def _score_stack(title: str, names, views, y, test_y, lam: float) -> str:
    oof = _stack_matrix(views, names, "oof")
    held = _stack_matrix(views, names, "test")
    pred = fit_predict(oof, y, held, lam)[1]
    text = f"{title}: lambda={lam:g}  n_base={len(names)}  " + _line(
        "test", binary_metrics(test_y, pred)
    )
    print(text, flush=True)
    return text


def _score_single(title: str, name: str, views, test_y) -> str:
    algo, feature = name.split(":", 1)
    views.ensure_oof(algo, feature)
    pred = views.ensure_test(name)
    text = f"{title}: " + _line("test", binary_metrics(test_y, pred))
    print(text, flush=True)
    return text


def run_narrow(train_pos, train_neg, test_pos, test_neg) -> None:
    train_seqs, train_y, test_seqs, test_y = load_split(
        train_pos, train_neg, test_pos, test_neg
    )
    folds = stratified_folds(train_y, 10, seed=0)
    inner = stratified_folds(train_y, 5, seed=1)
    print(
        f"Train {int(train_y.sum())}+{int((train_y == 0).sum())}  "
        f"test {int(test_y.sum())}+{int((test_y == 0).sum())}  folds=10  threshold=0.50",
        flush=True,
    )
    print(
        "StackDPPIV Table 3: ACC=0.891  Sn=0.857  Sp=0.925  MCC=0.784  AUC=0.961",
        flush=True,
    )
    views = _Views(_blocks(train_seqs, test_seqs), train_y, folds)

    print("\n=== step 1: four boosters ===", flush=True)
    oof_core = _stack_matrix(views, CORE, "oof")
    lam1, mcc1 = _choose_lambda(oof_core, train_y, inner)
    print(f"step1 train_MCC={mcc1:.3f}  lambda={lam1:g}", flush=True)

    print("\n=== step 2: first dipeptide ===", flush=True)
    step2_names = CORE + ("hgb:nterm",)
    oof_step2 = _stack_matrix(views, step2_names, "oof")
    lam2, mcc2 = _choose_lambda(oof_step2, train_y, inner)
    single = views.ensure_oof("hgb", "seq+dpc+nterm")
    mcc_single = _mcc(train_y, single)
    print(f"step2 stack train_MCC={mcc2:.3f}  lambda={lam2:g}", flush=True)
    print(f"step2 hgb:seq+dpc+nterm OOF_MCC={mcc_single:.3f}", flush=True)

    best_train = max(mcc1, mcc2, mcc_single)
    run_step3 = best_train < TRAIN_BAR
    print(
        f"best step1/step2 train_MCC={best_train:.3f}  bar={TRAIN_BAR:.2f}  "
        f"step3={'yes' if run_step3 else 'no'}",
        flush=True,
    )

    step3 = None
    if run_step3:
        print("\n=== step 3: ten encodings x five models, inner-fold selection ===", flush=True)
        names = []
        for feature in PAPER_FEATURES:
            for algo in PAPER_ALGOS:
                views.ensure_oof(algo, feature)
                names.append(f"{algo}:{feature}")
        oof_all = _stack_matrix(views, names, "oof")
        size, lam3, mcc3 = _select_subset(oof_all, train_y, inner)
        order = _rank_columns(oof_all, train_y)[:size]
        kept = [names[int(j)] for j in order]
        step3 = (size, lam3, mcc3, kept)
        print(
            f"step3 keeps k={size}  lambda={lam3:g}  train_MCC={mcc3:.3f}",
            flush=True,
        )
        print("step3 columns: " + ", ".join(kept), flush=True)

    print("\n=== independent test, one look each ===", flush=True)
    _score_stack("step1 four boosters", CORE, views, train_y, test_y, lam1)
    _score_stack("step2 four boosters + nterm", step2_names, views, train_y, test_y, lam2)
    _score_single("step2 hgb seq+dpc+nterm", "hgb:seq+dpc+nterm", views, test_y)
    if step3 is not None:
        size, lam3, _mcc3, kept = step3
        _score_stack(f"step3 selected k={size}", kept, views, train_y, test_y, lam3)


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
        run_narrow(
            paths["train_pos"], paths["train_neg"], paths["test_pos"], paths["test_neg"]
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
    run_narrow(args.train_pos, args.train_neg, args.test_pos, args.test_neg)


if __name__ == "__main__":
    main()
