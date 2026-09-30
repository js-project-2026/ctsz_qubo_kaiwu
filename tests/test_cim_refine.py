"""Local checks for CIM post-processing. No Kaiwu SDK and no network."""

import numpy as np

from qubo_model import (
    _energy,
    _ising_fields,
    compress_qubo_for_cim,
    local_refine_binary,
    quantize_ising_matrix,
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


def test_quantize_ising_stays_inside_signed_8bit():
    mat = np.array([[0.0, 0.2, -3.0], [0.2, 0.0, 0.01], [-3.0, 0.01, 0.0]])
    out = quantize_ising_matrix(mat, bits=8)
    assert out.min() >= -127
    assert out.max() <= 127
    assert np.all(np.diag(out) == 0)
