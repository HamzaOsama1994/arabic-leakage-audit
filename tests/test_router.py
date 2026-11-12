import numpy as np

from leakaudit.router import choose_tau, fit_temperature, novelty, selective, trust


def test_temperature_shrinks_overconfident_logits():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    noisy = np.where(rng.random(2000) < 0.7, y, 1 - y)  # 70% accurate
    logits = np.stack([(noisy == 0) * 10.0, (noisy == 1) * 10.0], axis=1)
    assert fit_temperature(logits, y) > 1


def test_novelty_zero_for_seen_rows_and_high_for_orthogonal():
    train = np.eye(3)[[0] * 10 + [1] * 10]
    assert np.allclose(novelty(np.eye(3)[[0]], train, k=10), 0)
    assert np.allclose(novelty(np.eye(3)[[2]], train, k=10), 1)


def test_trust_blocks_high_novelty_and_averages():
    p = [np.array([[0.2, 0.8], [0.9, 0.1]])] * 3
    t = trust(p, np.array([1, 0]), np.array([False, True]))
    assert np.isclose(t[0], 0.8) and t[1] == -np.inf


def test_tau_hits_target_coverage_and_selective():
    t = np.linspace(0, 1, 1000)
    acc = t >= choose_tau(t, 0.8)
    assert abs(acc.mean() - 0.8) < 0.01
    assert selective(acc, np.ones(1000, bool))["selective_accuracy"] == 1.0
