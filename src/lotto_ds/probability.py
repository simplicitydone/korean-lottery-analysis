"""probability.py — the probability foundations of 6/45, and the pension contrast.

This module is the *theory* the rest of the project measures reality against. It answers, from
first principles (no data), three questions:

1. **How likely is each prize tier?** — a hypergeometric draw (6 winners among 45, you pick 6).
2. **Why is the sum of a draw bell-shaped but a single position uniform?** — dependence + CLT.
3. **How does lotto differ from the pension lottery?** — the central teaching contrast:

   | property        | lotto 6/45                    | pension digits                 |
   |-----------------|-------------------------------|--------------------------------|
   | sampling        | without replacement           | with replacement (per digit)   |
   | within a draw   | numbers **dependent** (neg. corr) | positions **independent**  |
   | one slot's law  | not identically distributed   | each ~ Uniform{0..9}           |
   | sum's law       | ~Normal (CLT on dependent RVs)| ~Normal (CLT on independent)   |

Everything here is closed-form, so it doubles as the ground truth the empirical notebooks and the
statistical tests are compared against.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

from . import LOTTO_NUMBER_MAX, LOTTO_PICKS

WINNERS = LOTTO_PICKS          # 6 winning numbers
POOL = LOTTO_NUMBER_MAX        # 45


# ─────────────────────────── hypergeometric prize odds ──────────────────────

def match_probability(k: int, winners: int = WINNERS, pool: int = POOL, picks: int = LOTTO_PICKS) -> float:
    """P(exactly k of your `picks` match the `winners`) under a fair draw.

    Hypergeometric:  P(K=k) = C(winners,k)·C(pool-winners, picks-k) / C(pool, picks).
    """
    if k < 0 or k > picks:
        return 0.0
    return comb(winners, k) * comb(pool - winners, picks - k) / comb(pool, picks)


def expected_matches(picks: int = LOTTO_PICKS, winners: int = WINNERS, pool: int = POOL) -> float:
    """E[matches] = picks · winners / pool. For 6/45 this is 6·6/45 = 0.8 — the backtest baseline."""
    return picks * winners / pool


def prize_tier_table() -> list[dict]:
    """Odds for each Korean lotto prize tier (rank by matches; 2nd = 5 + bonus, approximated here
    as part of the 5-match mass for teaching). Returns k, probability, and '1 in N'."""
    rows = []
    for k in (6, 5, 4, 3):
        p = match_probability(k)
        rows.append({"matches": k, "probability": p, "one_in": (1 / p) if p else float("inf")})
    return rows


# ─────────────────────────── sum distribution (theory) ──────────────────────

@dataclass
class SumTheory:
    mean: float
    variance: float
    std: float
    min: int
    max: int


def sum_theory(picks: int = LOTTO_PICKS, pool: int = POOL) -> SumTheory:
    """Exact mean/variance of the sum of `picks` distinct draws from 1..pool (no replacement).

    Mean  = picks·(pool+1)/2.
    Var   = picks·(pool+1)·(pool-picks)/12   (the finite-population / no-replacement correction —
            the `(pool-picks)` factor is exactly the dependence term; with replacement it vanishes).
    For 6/45: mean = 138 (exact), variance = 897, std ≈ 29.95 — which matches the observed ~30.8
    closely, a clean confirmation that the sum behaves as sampling theory predicts.
    """
    mean = picks * (pool + 1) / 2
    var = picks * (pool + 1) * (pool - picks) / 12
    return SumTheory(mean=mean, variance=var, std=float(np.sqrt(var)), min=sum(range(1, picks + 1)),
                     max=sum(range(pool - picks + 1, pool + 1)))


# ─────────────────────────── independence vs dependence ─────────────────────

def within_draw_covariance(pool: int = POOL) -> float:
    """Covariance between two distinct positions of a without-replacement draw of indicators.

    For sampling without replacement, any two picks are *negatively* correlated:
        Cov(1{i drawn}, 1{j drawn}) < 0.
    This is the formal reason lotto numbers within a single draw are dependent — unlike pension
    digits, which are drawn independently with replacement (covariance 0).
    """
    p = LOTTO_PICKS / pool
    # hypergeometric indicator covariance: -p(1-p)/(N-1)
    return -p * (1 - p) / (pool - 1)


def simulate_position_marginal(pool: int = POOL, picks: int = LOTTO_PICKS, trials: int = 20000,
                               seed: int = 0) -> np.ndarray:
    """Empirical marginal of a *single sorted position* (e.g. the 1st/min number) — NOT uniform.

    Teaching point: while every number is equally likely to appear *somewhere* in the draw, a fixed
    sorted position (order statistic) has a skewed, non-uniform law. Contrast with pension, where
    each digit position is exactly Uniform{0..9}.
    """
    rng = np.random.default_rng(seed)
    mins = np.empty(trials, dtype=int)
    for t in range(trials):
        mins[t] = np.sort(rng.choice(np.arange(1, pool + 1), size=picks, replace=False))[0]
    return mins


def pension_digit_uniform_pmf() -> np.ndarray:
    """The exact law of one pension digit position: Uniform over {0..9} (independent, with repl.)."""
    return np.full(10, 1 / 10)
