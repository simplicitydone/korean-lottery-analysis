"""generator.py — 'prediction' generators for lotto and pension (honest by design).

Every lottery app ships a shiny "AI pick" button. So does this one — but with the honesty the rest
of the project earns: **these picks do not improve your odds** (see notebooks 04, 07, 09; every
combination is exactly equally likely). The generators are a demonstration of *how* such tools work
— frequency/hot/cold/contrarian weighting + a "jackpot profile" filter — not a betting edge.

Lotto: sample 6 of 45 under a strategy's weights, optionally keep only sets whose shape (sum, AC,
odd/even) matches the bulk of historical winners. Pension: sample a group (1–5) and 6 digits (0–9).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from . import LOTTO_NUMBER_MAX, LOTTO_PICKS
from .backtest import STRATEGIES as LOTTO_WEIGHTS
from .cleaning import WIN_COLS, load_clean
from .features import ac_value
from .pension import DIGIT_COLS, GROUPS, load_main

DISCLAIMER = ("생성된 조합은 재미와 교육용입니다. 어떤 방법도 당첨 확률을 높이지 않으며, "
              "모든 조합은 정확히 동일한 확률입니다 (노트북 04·07·09 참조).")

# Which of the backtest strategies to expose as named "methods" for lotto.
LOTTO_METHODS = {
    "빈도 (Frequency)": "frequency (all-time hot)",
    "핫 (Hot 50)": "hot (recent 50)",
    "콜드 (Cold)": "cold (overdue)",
    "역발상 (Contrarian)": "contrarian (rare)",
    "균형 (Balanced)": "random (uniform)",
}

# Jackpot-profile filter (the shape most historical winners share). Progressive relaxation
# guarantees we always return the requested count.
_JACKPOT_TIERS = [
    dict(sum_min=100, sum_max=180, ac_min=7, odds_ok={2, 3, 4}),
    dict(sum_min=90, sum_max=195, ac_min=6, odds_ok={2, 3, 4}),
    dict(sum_min=80, sum_max=210, ac_min=5, odds_ok={1, 2, 3, 4, 5}),
]


def generate_lotto(method: str = "빈도 (Frequency)", count: int = 5, seed: int | None = None,
                   draws: pd.DataFrame | None = None) -> list[dict]:
    if draws is None:
        draws = load_clean("draws")
    if method not in LOTTO_METHODS:
        raise ValueError(f"unknown method {method!r}")
    prior = np.sort(draws[WIN_COLS].to_numpy(dtype=int), axis=1)
    weights = LOTTO_WEIGHTS[LOTTO_METHODS[method]](prior)
    p = np.asarray(weights, dtype=float)
    p = p / p.sum()
    rng = np.random.default_rng(seed)

    results, seen = [], set()
    for tier in _JACKPOT_TIERS:
        for _ in range(8000):
            if len(results) >= count:
                break
            nums = tuple(sorted(int(x) for x in rng.choice(
                range(1, LOTTO_NUMBER_MAX + 1), size=LOTTO_PICKS, replace=False, p=p)))
            if nums in seen:
                continue
            s = sum(nums)
            odd = sum(1 for n in nums if n % 2)
            ac = ac_value(list(nums))
            if not (tier["sum_min"] <= s <= tier["sum_max"] and ac >= tier["ac_min"]
                    and odd in tier["odds_ok"]):
                continue
            seen.add(nums)
            results.append({"numbers": list(nums), "sum": s, "odd": odd, "ac": ac})
        if len(results) >= count:
            break
    return results[:count]


def generate_all_lotto(count: int = 5, seed: int = 0, draws: pd.DataFrame | None = None) -> dict:
    if draws is None:
        draws = load_clean("draws")
    return {
        "disclaimer": DISCLAIMER,
        "methods": {m: generate_lotto(m, count, seed + i, draws)
                    for i, m in enumerate(LOTTO_METHODS)},
    }


# ─────────────────────────── pension ────────────────────────────────────────

PENSION_METHODS = ["빈도 (Frequency)", "핫 (Hot)", "역발상 (Contrarian)", "균형 (Uniform)"]


def _pension_probs(df: pd.DataFrame, method: str):
    """Return (group_p[5], digit_p[6][10]) weight arrays for a method."""
    def col_counts(frame):
        gp = frame["group_no"].value_counts().reindex(GROUPS, fill_value=0).sort_index().to_numpy(float)
        dp = np.zeros((6, 10))
        for pi, c in enumerate(DIGIT_COLS):
            vc = pd.to_numeric(frame[c], errors="coerce").dropna().astype(int).value_counts()
            for d in range(10):
                dp[pi, d] = vc.get(d, 0)
        return gp, dp

    if method == "균형 (Uniform)":
        return np.ones(5), np.ones((6, 10))
    if method == "핫 (Hot)":
        gp, dp = col_counts(df.tail(40))
        return gp + 1, dp + 1
    gp, dp = col_counts(df)
    if method == "역발상 (Contrarian)":
        return 1.0 / (gp + 1), 1.0 / (dp + 1)
    return gp + 1, dp + 1  # 빈도 (Frequency)


def generate_pension(method: str = "빈도 (Frequency)", count: int = 5, seed: int | None = None,
                     df: pd.DataFrame | None = None) -> list[dict]:
    if df is None:
        df = load_main()
    if method not in PENSION_METHODS:
        raise ValueError(f"unknown method {method!r}")
    gp, dp = _pension_probs(df, method)
    gp = gp / gp.sum()
    dp = dp / dp.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)

    results, seen = [], set()
    for _ in range(3000):
        if len(results) >= count:
            break
        group = int(rng.choice(GROUPS, p=gp))
        digits = [int(rng.choice(range(10), p=dp[pi])) for pi in range(6)]
        key = (group, tuple(digits))
        if key in seen:
            continue
        seen.add(key)
        results.append({"group": group, "digits": digits})
    return results[:count]


def generate_all_pension(count: int = 5, seed: int = 0, df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = load_main()
    return {
        "disclaimer": DISCLAIMER,
        "methods": {m: generate_pension(m, count, seed + i, df)
                    for i, m in enumerate(PENSION_METHODS)},
    }
