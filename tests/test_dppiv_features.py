"""Dimensions and a few invariants for the added DPP-IV descriptors."""

import numpy as np

from dppiv_features import (
    AAI_DIM,
    APAAC_DIM,
    CTD_DIM,
    PAAC_DIM,
    aai_features,
    apaac_features,
    ctd_features,
    ctd_groups_partition_alphabet,
    paac_features,
)
from dppiv_logit import select_threshold


def test_ctd_groups_partition_and_charge_of_polyalanine():
    assert ctd_groups_partition_alphabet()
    X = ctd_features(["AAAA", "A"])
    assert X.shape == (2, CTD_DIM)
    # Charge is property index 10. Alanine is the neutral group, so composition is (0, 1, 0).
    charge = 10 * 21
    assert abs(X[0, charge] - 0.0) < 1e-12
    assert abs(X[0, charge + 1] - 1.0) < 1e-12
    assert abs(X[0, charge + 2] - 0.0) < 1e-12
    # A single residue has no transitions.
    assert X[1, charge + 3 : charge + 6].sum() == 0.0


def test_paac_and_apaac_are_compositions():
    paac = paac_features(["KP", "A"])
    apaac = apaac_features(["KPW"])
    aai = aai_features(["AAAA"])
    assert paac.shape == (2, PAAC_DIM)
    assert apaac.shape == (1, APAAC_DIM)
    assert aai.shape == (1, AAI_DIM)
    assert abs(paac[0].sum() - 1.0) < 1e-8
    assert abs(paac[1].sum() - 1.0) < 1e-8
    assert abs(apaac[0].sum() - 1.0) < 1e-8
    # Poly-alanine sits on alanine's own scales.
    assert aai[0, 9] == 0.0  # charge


def test_nterm_dipeptide_is_only_the_first_pair():
    from dppiv_features import NTERM_DIM, nterm_dipeptide_features

    X = nterm_dipeptide_features(["KPW", "A"])
    assert X.shape == (2, NTERM_DIM)
    # K is 8 and P is 12. PW must stay empty: only the first pair is marked.
    assert X[0, 8 * 20 + 12] == 1.0
    assert X[0].sum() == 1.0
    assert X[0, 12 * 20 + 18] == 0.0
    assert X[1].sum() == 0.0


def test_fingerprints_match_the_paper_widths():
    from dppiv_features import FP_DIMS, fingerprint_features

    for kind, width in FP_DIMS.items():
        X = fingerprint_features(["KPW", "A"], kind)
        assert X.shape == (2, width)


def test_threshold_moves_when_mcc_improves_off_one_half():
    y = np.array([1, 1, 1, 0, 0, 0])
    # At 0.5 every row is positive, so MCC is undefined or zero-class.
    # A cut at 0.70 keeps the three positives and drops the negatives.
    proba = np.array([0.95, 0.90, 0.80, 0.60, 0.40, 0.20])
    threshold, mcc = select_threshold(y, proba)
    assert threshold >= 0.60
    assert mcc > 0.9
