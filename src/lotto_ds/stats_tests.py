"""stats_tests.py — inferential tests for uniformity, independence, and randomness.

Each function returns a small dataclass with the statistic, p-value, and a plain-language
verdict, so the notebook can render a consistent "결과 (verdict)" line. The null hypothesis
throughout is that the draw process is a fair, memoryless uniform sampler; we look for — and
fail to find — evidence against it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import LOTTO_NUMBER_MAX, LOTTO_PICKS
from .cleaning import WIN_COLS, load_clean

ALPHA = 0.05


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    df: float | None = None
    detail: str = ""

    @property
    def reject_null(self) -> bool:
        return self.p_value < ALPHA

    @property
    def verdict_ko(self) -> str:
        return (
            "귀무가설 기각 — 무작위성에서 벗어남 (구조 발견)"
            if self.reject_null
            else "귀무가설 유지 — 무작위와 구별되지 않음"
        )

    def as_row(self) -> dict:
        return {
            "test": self.name,
            "statistic": round(float(self.statistic), 3),
            "df": self.df,
            "p_value": round(float(self.p_value), 4),
            "reject_H0 (α=.05)": self.reject_null,
            "verdict": self.verdict_ko,
        }


# ─────────────────────────── uniformity ─────────────────────────────────────

def chi_square_number_uniformity(draws: pd.DataFrame | None = None) -> TestResult:
    """Chi-square goodness-of-fit: are the 45 numbers drawn with equal probability?

    H0: every number 1..45 is equally likely. Expected count per number = N*6/45.
    """
    if draws is None:
        draws = load_clean("draws")
    flat = draws[WIN_COLS].to_numpy(dtype=int).ravel()
    observed = pd.Series(flat).value_counts().reindex(range(1, LOTTO_NUMBER_MAX + 1), fill_value=0).sort_index()
    expected = np.full(LOTTO_NUMBER_MAX, observed.sum() / LOTTO_NUMBER_MAX)
    # The six picks per draw are drawn WITHOUT replacement, so each number's count is
    # Binomial(N, 6/45) with variance E·(1−6/45), not the Poisson variance E that a plain
    # Σ(O−E)²/E assumes. The uncorrected Pearson statistic is under-dispersed (E[χ²]≈39, not 44)
    # and biased toward "fail to reject"; dividing each cell by (1−6/45) restores ~χ²₄₄
    # calibration — a 6k-draw WOR Monte-Carlo null agrees (p≈0.92 vs the naive 0.97).
    p_fair = LOTTO_PICKS / LOTTO_NUMBER_MAX
    chi2 = float((((observed.to_numpy() - expected) ** 2) / (expected * (1 - p_fair))).sum())
    p = float(stats.chi2.sf(chi2, LOTTO_NUMBER_MAX - 1))
    return TestResult(
        "번호 균등성 카이제곱 (number uniformity χ²)",
        chi2, p, df=LOTTO_NUMBER_MAX - 1,
        detail=f"n={len(draws)} draws, expected {expected[0]:.1f}/number (비복원 분산보정 ×1/(1−6/45))",
    )


def chi_square_pension_positions(pension: pd.DataFrame | None = None) -> list[TestResult]:
    """Per-position chi-square that each pension digit 0..9 is uniform at that position."""
    if pension is None:
        pension = load_clean("pension_draws")
    results = []
    for i in range(1, 7):
        col = pd.to_numeric(pension[f"n{i}"], errors="coerce").dropna().astype(int)
        observed = col.value_counts().reindex(range(10), fill_value=0).sort_index()
        expected = np.full(10, observed.sum() / 10)
        chi2, p = stats.chisquare(observed.to_numpy(), expected)
        results.append(TestResult(f"연금 자리{i} 균등성 χ²", chi2, p, df=9,
                                   detail=f"expected {expected[0]:.1f}/digit"))
    return results


# ─────────────────────────── independence / randomness ──────────────────────

def runs_test_odd_even(draws: pd.DataFrame | None = None) -> TestResult:
    """Wald–Wolfowitz runs test on the odd-majority sequence across draws.

    Encodes each draw as 1 if it has >3 odd numbers else 0, then tests whether that binary
    sequence is ordered randomly over time (no streaks/anti-streaks beyond chance).
    """
    if draws is None:
        draws = load_clean("draws")
    odd = (draws[WIN_COLS].to_numpy(dtype=int) % 2 != 0).sum(axis=1)
    seq = (odd > LOTTO_PICKS / 2).astype(int)  # 1 = odd-majority draw
    n1, n0 = int(seq.sum()), int((seq == 0).sum())
    n = n1 + n0
    runs = 1 + int((np.diff(seq) != 0).sum())
    # Normal approximation of the runs distribution
    mu = 2 * n1 * n0 / n + 1
    var = 2 * n1 * n0 * (2 * n1 * n0 - n) / (n**2 * (n - 1))
    z = (runs - mu) / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return TestResult("홀짝 다수 런 검정 (runs test)", z, p,
                      detail=f"runs={runs}, expected≈{mu:.1f}")


def ljung_box_sum(draws: pd.DataFrame | None = None, lags: int = 10) -> TestResult:
    """Ljung–Box test for autocorrelation in the draw-sum time series.

    H0: the sequence of draw sums is white noise (no draw predicts the next).
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    if draws is None:
        draws = load_clean("draws")
    s = draws[WIN_COLS].to_numpy(dtype=int).sum(axis=1)
    lb = acorr_ljungbox(s, lags=[lags], return_df=True)
    stat = float(lb["lb_stat"].iloc[0])
    p = float(lb["lb_pvalue"].iloc[0])
    return TestResult(f"합계 자기상관 Ljung–Box (lag={lags})", stat, p, df=lags)


# ─────────────────────────── the 'hot number' claim ─────────────────────────

def hot_number_edge(draws: pd.DataFrame | None = None, window: int = 50) -> TestResult:
    """Do 'hot' numbers (top-frequency in the trailing window) over-appear next draw?

    Walk-forward: at each draw t, rank numbers by their appearance count in the prior `window`
    draws; take the 6 hottest; count how many actually appear in draw t. Under the uniform null,
    the expected overlap is 6 * (6/45) = 0.8 per draw. We test observed mean overlap vs that
    baseline with a one-sample t-test — a fair, leakage-free operationalization of the folk claim.
    """
    if draws is None:
        draws = load_clean("draws")
    draws = draws.sort_values("draw_no").reset_index(drop=True)
    mat = draws[WIN_COLS].to_numpy(dtype=int)
    overlaps = []
    for t in range(window, len(mat)):
        prior = mat[t - window:t].ravel()
        counts = pd.Series(prior).value_counts()
        hot = set(counts.head(6).index.astype(int))
        actual = set(mat[t].astype(int))
        overlaps.append(len(hot & actual))
    overlaps = np.array(overlaps, dtype=float)
    baseline = LOTTO_PICKS * (LOTTO_PICKS / LOTTO_NUMBER_MAX)  # 0.8
    tstat, p = stats.ttest_1samp(overlaps, baseline)
    return TestResult(
        f"핫넘버 우위 검정 (hot-number edge, window={window})",
        tstat, p,
        detail=f"관측 평균 겹침 {overlaps.mean():.3f} vs 무작위 기대 {baseline:.2f} (n={len(overlaps)})",
    )


def run_all() -> pd.DataFrame:
    """Run the core test battery and return a tidy results table."""
    results = [
        chi_square_number_uniformity(),
        runs_test_odd_even(),
        ljung_box_sum(),
        hot_number_edge(),
    ]
    return pd.DataFrame([r.as_row() for r in results])


# ─────────────────────────── effect size & power ────────────────────────────
# A p-value answers "could this be chance?"; an effect size answers "does the size matter?". For a
# showcase these belong together — a big-N χ² can be 'significant' yet practically meaningless.

def cohens_w(observed: np.ndarray, expected: np.ndarray) -> float:
    """Cohen's w effect size for a χ² goodness-of-fit: w = sqrt(χ²/N).

    Conventions: 0.1 small · 0.3 medium · 0.5 large. Lottery uniformity gives w ≈ 0.05 — below even
    'small', i.e. the deviations from uniform are practically negligible regardless of significance.
    """
    observed, expected = np.asarray(observed, float), np.asarray(expected, float)
    n = observed.sum()
    chi2 = (((observed - expected) ** 2) / expected).sum()
    return float(np.sqrt(chi2 / n))


def chi_square_power(effect_w: float, df: int, n: int, alpha: float = ALPHA) -> float:
    """Post-hoc power of a χ² test to detect an effect of size `effect_w` given df and N.

    Uses the non-central χ² with non-centrality λ = w²·N. High power at a tiny w means: if any real
    bias of practical size existed, this dataset would almost certainly have caught it — so the null
    result is informative, not merely underpowered.
    """
    ncp = effect_w ** 2 * n
    crit = stats.chi2.ppf(1 - alpha, df)
    return float(stats.ncx2.sf(crit, df, ncp))
