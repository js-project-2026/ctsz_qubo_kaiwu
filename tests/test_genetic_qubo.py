"""Memetic GA on tiny QUBOs with a known minimum."""

import numpy as np

from qubo_model import _energy, solve_genetic_qubo


def test_negative_diagonal_turns_every_gene_on():
    Q = -np.eye(5)
    vec, energy = solve_genetic_qubo(Q, num_reads=2, seed=0)
    assert int(vec.sum()) == 5
    assert abs(energy - _energy(Q, vec)) < 1e-8
    assert energy < -4.9


def test_large_positive_coupling_keeps_one_gene():
    Q = np.array([[-0.2, 5.0], [5.0, -0.2]])
    vec, energy = solve_genetic_qubo(Q, num_reads=2, seed=1)
    assert int(vec.sum()) == 1
    assert energy < 0
    assert abs(energy - _energy(Q, vec)) < 1e-8
