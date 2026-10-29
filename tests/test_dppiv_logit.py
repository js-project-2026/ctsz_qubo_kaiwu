"""Feature map, logistic loss, and metrics for the DPP-IV model."""

import numpy as np

from dppiv_logit import (
    binary_metrics,
    feature_matrix,
    fit_predict,
    sequence_features,
)


def test_composition_sums_to_one_and_second_residue_is_marked():
    X = sequence_features(["KP", "A"])
    assert X.shape == (2, 80)
    assert abs(X[0, :20].sum() - 1.0) < 1e-12
    assert abs(X[1, :20].sum() - 1.0) < 1e-12
    # Second residue of "A" is absent.
    assert X[1, 40:60].sum() == 0.0
    # Second residue of "KP" is P. P is index 12 in ACDEFGHIKLMNPQRSTVWY.
    assert X[0, 40 + 12] == 1.0


def test_length_feature_is_log_length():
    X = feature_matrix(["AA", "AAAA"], "len")
    assert X.shape == (2, 1)
    assert abs(X[1, 0] - np.log(4)) < 1e-12


def test_separated_residues_are_classified():
    seqs = ["A"] * 20 + ["W"] * 20
    y = np.array([1] * 20 + [0] * 20)
    X = feature_matrix(seqs, "seq")
    train_p, test_p, _ = fit_predict(X, y, X, lam=0.01)
    metrics = binary_metrics(y, test_p)
    assert metrics["ACC"] == 1.0
    assert metrics["AUC"] == 1.0
    assert metrics["MCC"] == 1.0
    assert train_p[0] > 0.9
    assert train_p[-1] < 0.1


def test_auc_of_a_perfect_ranking_and_a_tie():
    y = np.array([1, 1, 0, 0])
    assert abs(binary_metrics(y, np.array([0.9, 0.8, 0.2, 0.1]))["AUC"] - 1.0) < 1e-12
    tied = binary_metrics(y, np.array([0.5, 0.5, 0.5, 0.5]))
    assert abs(tied["AUC"] - 0.5) < 1e-12
    # Every score is on the positive side of the 0.5 cut.
    assert tied["ACC"] == 0.5
    assert tied["Sn"] == 1.0
    assert tied["Sp"] == 0.0
