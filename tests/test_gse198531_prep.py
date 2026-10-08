"""Gene and age rules for the GSE198531 preprocessing script."""

import numpy as np

from gse198531_qubo import AGE_ORDER, age_label, age_ranks, technical_gene


def test_technical_genes_are_the_three_classes():
    assert technical_gene("mt-Atp6") == "mt"
    assert technical_gene("Rpl13") == "ribo"
    assert technical_gene("Rps6") == "ribo"
    assert technical_gene("Gm28437") == "Gm"
    assert technical_gene("Calb1") is None
    assert technical_gene("Tyrobp") is None


def test_a2m_is_not_an_age_and_ranks_are_even():
    assert age_label("inflammation: A2M") is None
    assert age_label("age: E12.5") == "E12.5"
    assert age_label("stage: Adult") == "Adult"
    y = age_ranks(np.asarray(AGE_ORDER))
    assert y.tolist() == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
