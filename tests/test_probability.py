"""Tests for the closed-form probability foundations (module has no data dependency)."""

from math import comb, isclose

from src.lotto_ds import probability as pb


def test_match_probabilities_sum_to_one():
    total = sum(pb.match_probability(k) for k in range(0, 7))
    assert isclose(total, 1.0, abs_tol=1e-12)


def test_expected_matches_is_point_eight():
    assert isclose(pb.expected_matches(), 0.8)


def test_jackpot_odds():
    assert isclose(pb.match_probability(6), 1 / comb(45, 6))
    assert comb(45, 6) == 8_145_060


def test_sum_theory_mean_is_138():
    th = pb.sum_theory()
    assert th.mean == 138
    assert th.min == 21 and th.max == 255
    assert 29 < th.std < 31  # matches the observed ~30.8


def test_within_draw_covariance_negative():
    # sampling without replacement => negatively correlated indicators (dependence)
    assert pb.within_draw_covariance() < 0


def test_position_marginal_not_uniform():
    # the sorted-minimum is skewed toward small values, unlike a pension digit
    mins = pb.simulate_position_marginal(trials=4000, seed=1)
    assert mins.mean() < 12  # far below the 23 midpoint => non-uniform
