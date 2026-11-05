import numpy as np

from xaiclinic.stability import mean_jaccard, mean_kendall, selection_frequency, topk_sets


def test_identical_rankings_are_perfectly_stable():
    imp = np.tile(np.array([5.0, 4, 3, 2, 1]), (6, 1))
    assert mean_jaccard(topk_sets(imp, 2)) == 1.0
    assert abs(mean_kendall(imp) - 1.0) < 1e-12


def test_disjoint_topk_gives_zero_jaccard():
    imp = np.array([[3.0, 2, 1, 0], [0, 1, 2, 3]])
    assert mean_jaccard(topk_sets(imp, 1)) == 0.0
    assert mean_kendall(imp) < -0.9


def test_selection_frequency_sums_to_k():
    rng = np.random.default_rng(0)
    imp = rng.normal(size=(20, 8))
    f = selection_frequency(topk_sets(imp, 3), 8)
    assert abs(f.sum() - 3) < 1e-9
