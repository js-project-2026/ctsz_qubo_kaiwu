"""Memetic GA on tiny QUBOs with a known minimum."""

import numpy as np

from qubo_model import _energy, mqlib_qubo_terms, solve_genetic_qubo


def test_negative_diagonal_turns_every_gene_on():
    Q = -np.eye(5)
    vec, energy = solve_genetic_qubo(Q, num_reads=2, seed=0)
    assert int(vec.sum()) == 5
    assert abs(energy - _energy(Q, vec)) < 1e-8
    assert energy < -4.9


def test_mqlib_terms_match_negated_energy():
    Q = np.array([[5.0, -6.0, 0.0], [-6.0, 3.0, -1.0], [0.0, -1.0, 1.0]])
    terms = {(i, j): w for i, j, w in mqlib_qubo_terms(Q)}
    vec = np.array([1, 1, 0])

    def mqlib_objective(x):
        total = 0.0
        for i in range(3):
            if x[i]:
                total += terms[(i + 1, i + 1)]
        for i, j in ((1, 2), (2, 3)):
            if x[i - 1] and x[j - 1]:
                total += terms[(i, j)]
        return total

    assert abs(mqlib_objective(vec) + _energy(Q, vec)) < 1e-8


def test_large_positive_coupling_keeps_one_gene():
    Q = np.array([[-0.2, 5.0], [5.0, -0.2]])
    vec, energy = solve_genetic_qubo(Q, num_reads=2, seed=1)
    assert int(vec.sum()) == 1
    assert energy < 0
    assert abs(energy - _energy(Q, vec)) < 1e-8
