"""bayesian.py — Beta-Binomial estimation of each number's true appearance probability.

A frequentist χ² asks "are the counts consistent with uniform?" (yes). A Bayesian asks the more
useful question for a would-be gambler: **"given the data, what is each number's probability, and
how sure are we?"** With a conjugate Beta prior on a Bernoulli/Binomial likelihood, the posterior is
closed-form — no MCMC needed — which makes it an ideal teaching example.

Model (per number i, across N draws):
    appearances k_i ~ Binomial(N, θ_i)          θ_i = P(number i appears in a given draw)
    prior     θ_i ~ Beta(α, β)                   (α=β chosen so the prior mean = 6/45)
    posterior θ_i ~ Beta(α + k_i, β + N − k_i)   (conjugacy)

The null "fair" value is θ = 6/45 ≈ 0.1333. The teaching payoff: every number's 95% **credible
interval** comfortably contains 6/45, and the posterior means are *shrunk* from the noisy empirical
rates toward the prior — visually demystifying "regression to the mean" and why hot/cold streaks are
mirages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import LOTTO_NUMBER_MAX, LOTTO_PICKS
from .cleaning import WIN_COLS, load_clean

FAIR_THETA = LOTTO_PICKS / LOTTO_NUMBER_MAX  # 6/45 ≈ 0.1333


def _prior_from_mean(mean: float, strength: float) -> tuple[float, float]:
    """Beta(α, β) with the given mean and concentration `strength` = α+β (prior 'pseudo-draws')."""
    alpha = mean * strength
    beta = strength - alpha
    return alpha, beta


def posterior_table(draws: pd.DataFrame | None = None, prior_strength: float = 45.0,
                    cred_mass: float = 0.95) -> pd.DataFrame:
    """Per-number Beta-Binomial posterior with credible interval.

    `prior_strength` is the prior's weight in equivalent draws (α+β). 45 is a mild, honest prior
    centered at the fair rate 6/45. Returns one row per number 1..45.
    """
    if draws is None:
        draws = load_clean("draws")
    n_draws = len(draws)
    mat = draws[WIN_COLS].to_numpy(dtype=int)

    alpha0, beta0 = _prior_from_mean(FAIR_THETA, prior_strength)
    lo_q, hi_q = (1 - cred_mass) / 2, 1 - (1 - cred_mass) / 2

    rows = []
    for num in range(1, LOTTO_NUMBER_MAX + 1):
        appearances = int((mat == num).any(axis=1).sum())  # draws in which the number appeared
        a = alpha0 + appearances
        b = beta0 + (n_draws - appearances)
        post = stats.beta(a, b)
        rows.append({
            "number": num,
            "appearances": appearances,
            "empirical_rate": appearances / n_draws,
            "posterior_mean": a / (a + b),
            "ci_lo": float(post.ppf(lo_q)),
            "ci_hi": float(post.ppf(hi_q)),
            # probability the number is "hotter than fair", P(θ > 6/45 | data)
            "p_above_fair": float(post.sf(FAIR_THETA)),
        })
    out = pd.DataFrame(rows)
    out["ci_contains_fair"] = (out["ci_lo"] <= FAIR_THETA) & (FAIR_THETA <= out["ci_hi"])
    return out


def summary(draws: pd.DataFrame | None = None, prior_strength: float = 45.0) -> dict:
    """Headline numbers for the notebook / web app."""
    tbl = posterior_table(draws, prior_strength)
    return {
        "fair_theta": FAIR_THETA,
        "n_numbers": len(tbl),
        "ci_contains_fair_count": int(tbl["ci_contains_fair"].sum()),
        "shrinkage_note": "posterior means pulled from empirical toward the 6/45 prior",
        "max_p_above_fair": float(tbl["p_above_fair"].max()),
        "min_p_above_fair": float(tbl["p_above_fair"].min()),
    }
