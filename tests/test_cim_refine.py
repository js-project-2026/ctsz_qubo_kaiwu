"""Local checks for CIM post-processing. No Kaiwu SDK and no network."""

import numpy as np

from qubo_model import (
    _energy,
    _ising_fields,
    compress_qubo_for_cim,
    ising_matrix_from_qubo,
    local_refine_binary,
    quantized_ising_signal,
    quantize_ising_matrix,
    split_cim_masks,
)


def test_refine_leaves_the_empty_mask_when_diagonal_is_negative():
    Q = np.array([[-1.0, 0.2], [0.2, -0.4]])
    x, energy = local_refine_binary(Q, np.zeros(2, dtype=int))
    assert energy < -1e-9
    assert int(x.sum()) >= 1
    assert abs(energy - _energy(Q, x)) < 1e-9


def test_refine_turns_off_a_costly_pair():
    Q = np.array([[-0.2, 5.0], [5.0, -0.2]])
    x, energy = local_refine_binary(Q, np.ones(2, dtype=int))
    assert int(x.sum()) == 1
    assert energy < 0
    assert abs(energy - _energy(Q, x)) < 1e-9


def test_refine_matches_brute_force_one_bit_delta():
    rng = np.random.default_rng(1)
    Q = rng.normal(size=(6, 6))
    Q = 0.5 * (Q + Q.T)
    x = rng.integers(0, 2, size=6)
    h = Q @ x.astype(float)
    diag = np.diag(Q)
    for i in range(6):
        trial = x.copy()
        trial[i] = 1 - trial[i]
        xi = float(x[i])
        coupled = h[i] - diag[i] * xi
        d = 1.0 - 2.0 * xi
        delta = d * (diag[i] + 2.0 * coupled)
        assert abs(delta - (_energy(Q, trial) - _energy(Q, x))) < 1e-8


def test_dense_qubo_needs_a_threshold_before_8bit_ising_keeps_a_field():
    rng = np.random.default_rng(0)
    n = 80
    noise = np.abs(rng.normal(size=(n, n)))
    R = np.exp(-noise) * 0.25
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 0.0)
    I = rng.random(n) * 0.35
    Q = (1 - 0.3125) * R / 49.0 - 0.3125 * np.diag(I)
    Q = 0.5 * (Q + Q.T)

    h_dense, _ = _ising_fields(Q)
    peak = np.max(np.abs(h_dense))
    assert int(np.sum(np.rint(h_dense / peak * 127) < 0)) == 0

    Qk, info = compress_qubo_for_cim(Q, bits=8, min_negative_fields=1)
    assert info["tau"] > 0
    assert info["negative_fields"] >= 1
    assert info["edges_kept"] < info["edges_total"]
    h, _ = _ising_fields(Qk)
    j_peak = np.max(np.abs(Qk - np.diag(np.diag(Qk)))) / 4.0
    scale = 127.0 / max(np.max(np.abs(h)), j_peak)
    assert int(np.sum(np.rint(h * scale) < 0)) >= 1


def _improving_moves(Q, x):
    Q = 0.5 * (Q + Q.T)
    x = np.asarray(x, dtype=float)
    h = Q @ x
    diag = np.diag(Q)
    n_move = 0
    for i in range(x.size):
        coupled = h[i] - diag[i] * x[i]
        d = 1.0 - 2.0 * x[i]
        if d * (diag[i] + 2.0 * coupled) < -1e-12:
            n_move += 1
    return n_move


def test_default_refine_reaches_a_local_minimum():
    rng = np.random.default_rng(2)
    Q = rng.normal(size=(12, 12))
    Q = 0.5 * (Q + Q.T)
    x0 = rng.integers(0, 2, size=12)
    x, energy = local_refine_binary(Q, x0)
    assert _improving_moves(Q, x) == 0
    assert abs(energy - _energy(Q, x)) < 1e-8
    assert local_refine_binary.converged


def test_split_labels_hardware_genes_separately_from_refine():
    raw = np.array([1, 1, 0, 0])
    refined = np.array([1, 0, 1, 0])
    split = split_cim_masks(raw, refined)
    assert split["hardware"] == [0, 1]
    assert split["kept"] == [0]
    assert split["added"] == [2]
    assert split["removed"] == [1]


def test_range_fit_scores_the_ising_matrix_it_will_upload():
    rng = np.random.default_rng(0)
    n = 40
    noise = np.abs(rng.normal(size=(n, n)))
    R = np.exp(-noise) * 0.25
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 0.0)
    I = rng.random(n) * 0.35
    Q = (1 - 0.3125) * R / 49.0 - 0.3125 * np.diag(I)
    Q = 0.5 * (Q + Q.T)

    def to_ising_boosted(matrix):
        ising = ising_matrix_from_qubo(matrix)
        ising[:n, :n] *= 50.0
        return ising

    Qk, info = compress_qubo_for_cim(
        Q, bits=8, min_negative_fields=1, to_ising=to_ising_boosted
    )
    neg, nz = quantized_ising_signal(to_ising_boosted(Qk), n, bits=8)
    plain_neg, plain_nz = quantized_ising_signal(ising_matrix_from_qubo(Qk), n, bits=8)
    assert (info["negative_fields"], info["quantized_edges"]) == (neg, nz)
    assert (neg, nz) != (plain_neg, plain_nz)
    assert neg >= 1 and nz >= 1


def test_range_fit_rejects_a_single_surviving_coupling():
    """tau=0 with one quantized edge must lose to a cut that keeps more edges."""
    rng = np.random.default_rng(3)
    n = 30
    noise = np.abs(rng.normal(size=(n, n)))
    R = np.exp(-noise) * 0.25
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 0.0)
    I = rng.random(n) * 0.35
    Q = (1 - 0.3125) * R / 49.0 - 0.3125 * np.diag(I)
    Q = 0.5 * (Q + Q.T)

    def to_ising_shrink(matrix):
        ising = ising_matrix_from_qubo(matrix)
        ising[:n, :n] *= 0.15
        return ising

    dense_neg, dense_nz = quantized_ising_signal(to_ising_shrink(Q), n, bits=8)
    Qk, info = compress_qubo_for_cim(
        Q, bits=8, min_negative_fields=1, to_ising=to_ising_shrink
    )
    assert info["quantized_edges_at_zero"] == dense_nz
    assert info["quantized_edges"] > dense_nz
    assert info["quantized_edges"] > 1
    assert info["negative_fields"] >= 1
    assert info["tau"] > 0
    neg, nz = quantized_ising_signal(to_ising_shrink(Qk), n, bits=8)
    assert (neg, nz) == (info["negative_fields"], info["quantized_edges"])


def test_quantize_ising_stays_inside_signed_8bit():
    mat = np.array([[0.0, 0.2, -3.0], [0.2, 0.0, 0.01], [-3.0, 0.01, 0.0]])
    out = quantize_ising_matrix(mat, bits=8)
    assert out.min() >= -127
    assert out.max() <= 127
    assert np.all(np.diag(out) == 0)
