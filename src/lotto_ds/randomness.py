"""randomness.py — a 'is this stream actually random?' test battery.

Casinos, RNG auditors, and cryptographers don't eyeball a sequence and call it random — they run a
*battery* of tests, each sensitive to a different kind of structure. This module assembles a
teaching-sized version of that toolkit and points it at the lottery draw stream:

- **Shannon entropy** — is the per-number information content maximal (log2 45 bits)?
- **Two-sample KS (drift)** — do the first and second halves of history share one distribution?
- **Anderson–Darling** — is the draw-sum ~Normal, as the CLT predicts for a fair draw?
- **ADF stationarity** — is the draw-sum series stable (no unit root / drift)?
- **Serial / lag-1 parity** — does one draw's parity predict the next (Markov-ish dependence)?
- **Spectral (permutation)** — any periodic component beyond what shuffling produces?

A subtlety worth teaching: these tests point in **different directions**. For most, *high* p means
"consistent with random". For ADF, it is the opposite — its null is "has a unit root (non-random
drift)", so a *low* p (reject) is the random-friendly outcome. `run_battery()` resolves this per
test and reports one honest "무작위 부합 (random-consistent)" column, rather than a raw reject flag
that would read backwards for ADF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from . import LOTTO_NUMBER_MAX
from .cleaning import WIN_COLS, load_clean
from .stats_tests import TestResult

# Tests whose *random-consistent* outcome is a LOW p-value (reject their null). Everything else is
# random-consistent when p is HIGH (fail to reject).
_RANDOM_WHEN_REJECT = {"ADF 정상성 (stationarity)"}


def _flat_numbers(draws: pd.DataFrame) -> np.ndarray:
    return draws[WIN_COLS].to_numpy(dtype=int).ravel()


def _sums(draws: pd.DataFrame) -> np.ndarray:
    return draws[WIN_COLS].to_numpy(int).sum(axis=1).astype(float)


def shannon_entropy(draws: pd.DataFrame | None = None) -> TestResult:
    """Entropy of the number distribution vs the theoretical maximum log2(45).

    The 'statistic' is the efficiency ratio (0..1); a uniform source hits 1. The p-value is the
    χ² uniformity p, so it slots into the same verdict logic.
    """
    if draws is None:
        draws = load_clean("draws")
    counts = pd.Series(_flat_numbers(draws)).value_counts().reindex(
        range(1, LOTTO_NUMBER_MAX + 1), fill_value=0).sort_index().to_numpy()
    probs = counts / counts.sum()
    nz = probs[probs > 0]
    ent = float(-(nz * np.log2(nz)).sum())
    max_ent = float(np.log2(LOTTO_NUMBER_MAX))
    _, p = stats.chisquare(counts, np.full(LOTTO_NUMBER_MAX, counts.sum() / LOTTO_NUMBER_MAX))
    return TestResult("샤논 엔트로피 효율 (entropy efficiency)", ent / max_ent, p,
                      detail=f"{ent:.4f} / {max_ent:.4f} bits ({ent/max_ent*100:.2f}% of max)")


def ks_drift(draws: pd.DataFrame | None = None) -> TestResult:
    """Two-sample KS: do the draw-sums of the first half and second half share one distribution?

    A valid continuous use of KS (sums have few ties), and a direct test of *temporal drift* — if the
    process changed over 20+ years, the halves would diverge. Fair process → high p.
    """
    if draws is None:
        draws = load_clean("draws")
    s = _sums(draws)
    mid = len(s) // 2
    stat, p = stats.ks_2samp(s[:mid], s[mid:])
    return TestResult("전·후반 분포 동일성 (two-sample KS drift)", stat, p,
                      detail=f"first {mid} vs last {len(s)-mid} draws")


def anderson_darling_sum(draws: pd.DataFrame | None = None) -> TestResult:
    """Anderson–Darling normality of the draw-sum series (CLT predicts ~Normal).

    Returns A² with a real p-value via the Stephens (1974) small-sample formula (reject ⇒ not
    Normal). A fair lottery should NOT reject.
    """
    if draws is None:
        draws = load_clean("draws")
    x = np.sort(_sums(draws))
    n = len(x)
    # A² computed directly (matches scipy.stats.anderson to 1e-6) so we avoid scipy's deprecated
    # `.critical_values` (removed in SciPy 1.19) and can report a real p, not a 0.025/0.5 placeholder.
    z = (x - x.mean()) / x.std(ddof=1)
    cdf = np.clip(stats.norm.cdf(z), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    a2 = float(-n - np.sum((2 * i - 1) * (np.log(cdf) + np.log(1 - cdf[::-1]))) / n)
    a2s = a2 * (1 + 0.75 / n + 2.25 / n**2)      # Stephens (1974) small-sample correction
    if a2s < 0.2:
        p = 1 - np.exp(-13.436 + 101.14 * a2s - 223.73 * a2s**2)
    elif a2s < 0.34:
        p = 1 - np.exp(-8.318 + 42.796 * a2s - 59.938 * a2s**2)
    elif a2s < 0.6:
        p = np.exp(0.9177 - 4.279 * a2s - 1.38 * a2s**2)
    elif a2s < 10:
        p = np.exp(1.2937 - 5.709 * a2s + 0.0186 * a2s**2)
    else:
        p = 0.0
    return TestResult("Anderson–Darling 정규성 (sum normality)", a2, float(p),
                      detail=f"A²={a2:.3f}, p={p:.3f} (Stephens 1974)")


def adf_stationarity(draws: pd.DataFrame | None = None) -> TestResult:
    """Augmented Dickey–Fuller: is the draw-sum series stationary (no unit root)?

    H0 = has a unit root (non-stationary drift). Rejecting is the random-friendly outcome — handled
    by `_RANDOM_WHEN_REJECT`.
    """
    if draws is None:
        draws = load_clean("draws")
    stat, p, *_ = adfuller(_sums(draws), autolag="AIC")
    return TestResult("ADF 정상성 (stationarity)", stat, p, detail="H0=단위근(비정상). 기각=정상.")


def serial_parity(draws: pd.DataFrame | None = None) -> TestResult:
    """Lag-1 serial test: does one draw's odd-count parity predict the next? (χ² on a 2×2 table)."""
    if draws is None:
        draws = load_clean("draws")
    odd = (draws[WIN_COLS].to_numpy(int) % 2 != 0).sum(axis=1)
    parity = (odd > 3).astype(int)
    table = pd.crosstab(pd.Series(parity[:-1], name="t"), pd.Series(parity[1:], name="t1"))
    chi2, p, *_ = stats.chi2_contingency(table)
    return TestResult("직렬 종속 (lag-1 parity χ²)", chi2, p)


def spectral_permutation(draws: pd.DataFrame | None = None, n_perm: int = 2000,
                         seed: int = 0) -> TestResult:
    """Permutation spectral test: is the strongest periodogram peak of the sum series real?

    We measure the max normalized FFT ordinate `g`, then shuffle the series `n_perm` times (killing
    any time order) and see how often a shuffle produces an equal-or-larger peak. That fraction is an
    exact, assumption-free p-value — a clean teaching example of permutation testing, robust where
    the analytic Fisher-g approximation is anti-conservative.
    """
    if draws is None:
        draws = load_clean("draws")
    s = _sums(draws)
    s = s - s.mean()

    def peak(x):
        power = np.abs(np.fft.rfft(x))[1:] ** 2
        return power.max() / power.sum()

    observed = peak(s)
    rng = np.random.default_rng(seed)
    ge = sum(peak(rng.permutation(s)) >= observed for _ in range(n_perm))
    p = (ge + 1) / (n_perm + 1)
    return TestResult("스펙트럼 최대 봉우리 (permutation spectral)", float(observed), p,
                      detail=f"peak carries {observed*100:.2f}% of power; {n_perm} permutations")


def run_battery(draws: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run all randomness tests and return a tidy table with an honest random-consistency column."""
    if draws is None:
        draws = load_clean("draws")
    tests = [
        shannon_entropy(draws), ks_drift(draws), anderson_darling_sum(draws),
        adf_stationarity(draws), serial_parity(draws), spectral_permutation(draws, n_perm=10000),
    ]
    rows = []
    for t in tests:
        random_when_reject = t.name in _RANDOM_WHEN_REJECT
        consistent = t.reject_null if random_when_reject else not t.reject_null
        rows.append({
            "test": t.name,
            "statistic": round(float(t.statistic), 3),
            "p_value": round(float(t.p_value), 4),
            "random_consistent": consistent,
            "detail": t.detail,
        })
    return pd.DataFrame(rows)
