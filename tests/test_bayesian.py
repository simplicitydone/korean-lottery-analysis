"""Tests for the Beta-Binomial posterior estimation."""

import numpy as np
import pandas as pd

from src.lotto_ds import bayesian as by


def _synthetic_fair_draws(n=800, seed=0):
    rng = np.random.default_rng(seed)
    rows = [np.sort(rng.choice(range(1, 46), size=6, replace=False)) for _ in range(n)]
    return pd.DataFrame(rows, columns=[f"win{i}" for i in range(1, 7)])


def test_table_shape_and_fair_theta():
    tbl = by.posterior_table(_synthetic_fair_draws())
    assert len(tbl) == 45
    assert abs(by.FAIR_THETA - 6 / 45) < 1e-9


def test_credible_interval_coverage_on_fair_data():
    # With fair data, the vast majority of 95% credible intervals should contain 6/45.
    tbl = by.posterior_table(_synthetic_fair_draws(n=1000, seed=3))
    assert tbl["ci_contains_fair"].mean() >= 0.85


def test_posterior_is_shrunk_between_empirical_and_prior():
    # Each posterior mean lies between the empirical rate and the prior mean (shrinkage).
    tbl = by.posterior_table(_synthetic_fair_draws())
    for _, r in tbl.iterrows():
        lo, hi = sorted([r["empirical_rate"], by.FAIR_THETA])
        assert lo - 1e-9 <= r["posterior_mean"] <= hi + 1e-9


def test_stronger_prior_shrinks_more():
    draws = _synthetic_fair_draws()
    weak = by.posterior_table(draws, prior_strength=10)
    strong = by.posterior_table(draws, prior_strength=500)
    spread_weak = weak["posterior_mean"].max() - weak["posterior_mean"].min()
    spread_strong = strong["posterior_mean"].max() - strong["posterior_mean"].min()
    assert spread_strong < spread_weak
