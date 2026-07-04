"""pension.py — 연금복권 720+ analysis, prize rules, and ticket checking.

The pension lottery is a useful counterpoint to lotto (see notebook 02): a ticket is one **group**
(조, 1–5) plus **6 independent digits** (each 0–9), drawn *with replacement*. Prizes are tiered by
how many **trailing** digits match the winning number (plus the group for 1st prize):

    1등  group + all 6 digits           1 / 5,000,000
    2등  any group + all 6 digits       1 / 1,000,000
    3등  last 5 digits                  1 / 100,000
    4등  last 4 digits                  1 / 10,000
    5등  last 3 digits                  1 / 1,000
    6등  last 2 digits                  1 / 100
    7등  last 1 digit                   1 / 10
    보너스  bonus number, all 6 digits   (separate draw, group-agnostic)

This module derives group/digit statistics fresh from the clean data, exposes the odds table, and
checks any ticket against a real historical draw — the honest counterpart to lotto's checker.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cleaning import load_clean

DIGIT_COLS = [f"n{i}" for i in range(1, 7)]
GROUPS = [1, 2, 3, 4, 5]

# (label, odds "1 in N", approx prize KRW description) — standard 720+ structure.
PRIZE_TIERS = [
    {"tier": "1등", "match": "조 + 6자리", "one_in": 5_000_000, "prize": "매월 700만원 × 20년"},
    {"tier": "2등", "match": "6자리 (조 무관)", "one_in": 1_000_000, "prize": "매월 100만원 × 10년"},
    {"tier": "3등", "match": "뒤 5자리", "one_in": 100_000, "prize": "100만원"},
    {"tier": "4등", "match": "뒤 4자리", "one_in": 10_000, "prize": "10만원"},
    {"tier": "5등", "match": "뒤 3자리", "one_in": 1_000, "prize": "5만원"},
    {"tier": "6등", "match": "뒤 2자리", "one_in": 100, "prize": "5천원"},
    {"tier": "7등", "match": "뒤 1자리", "one_in": 10, "prize": "1천원"},
]


def load_main() -> pd.DataFrame:
    df = load_clean("pension_draws")
    for c in ["group_no", *DIGIT_COLS]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df.sort_values("draw_no").reset_index(drop=True)


def load_bonus() -> pd.DataFrame:
    df = load_clean("pension_bonus")
    for c in DIGIT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df.sort_values("draw_no").reset_index(drop=True)


# ─────────────────────────── statistics ─────────────────────────────────────

def group_frequency(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_main()
    counts = df["group_no"].value_counts().reindex(GROUPS, fill_value=0).sort_index()
    n = int(counts.sum())
    return pd.DataFrame({
        "group": GROUPS,
        "count": counts.astype(int).to_numpy(),
        "share": (counts / n).to_numpy(),
        "expected": n / len(GROUPS),
    })


def digit_position_frequency(df: pd.DataFrame | None = None) -> np.ndarray:
    """(6, 10) matrix of digit counts per position."""
    if df is None:
        df = load_main()
    mat = np.zeros((6, 10), dtype=int)
    for pi, col in enumerate(DIGIT_COLS):
        vc = pd.to_numeric(df[col], errors="coerce").dropna().astype(int).value_counts()
        for d in range(10):
            mat[pi, d] = int(vc.get(d, 0))
    return mat


def digit_sum_series(df: pd.DataFrame | None = None) -> np.ndarray:
    if df is None:
        df = load_main()
    return df[DIGIT_COLS].to_numpy(dtype=float).sum(axis=1)


def latest_draws(n: int = 10, df: pd.DataFrame | None = None) -> list[dict]:
    if df is None:
        df = load_main()
    out = df.tail(n).iloc[::-1]
    return [
        {"draw_no": int(r.draw_no), "draw_date": str(r.draw_date),
         "group": int(r.group_no), "digits": [int(r[c]) for c in DIGIT_COLS]}
        for _, r in out.iterrows()
    ]


# ─────────────────────────── prize checking ─────────────────────────────────

def _trailing_match(a: list[int], b: list[int]) -> int:
    """Number of matching trailing digits between two 6-digit sequences."""
    k = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x == y:
            k += 1
        else:
            break
    return k


@dataclass
class CheckResult:
    tier: str
    matched_trailing: int
    group_match: bool
    one_in: int | None
    prize: str
    win_group: int
    win_digits: list[int]
    draw_no: int


def check_ticket(group: int, digits: list[int], draw_no: int | None = None,
                 df: pd.DataFrame | None = None) -> CheckResult:
    """Check a ticket (group 1–5, 6 digits) against a real draw (latest if unspecified).

    Returns the prize tier it would have won. This is the honest 'what would you have won?' — every
    ticket has identical odds regardless of how it was chosen.
    """
    if df is None:
        df = load_main()
    if not (1 <= int(group) <= 5):
        raise ValueError("조는 1~5 사이여야 합니다")
    digits = [int(d) for d in digits]
    if len(digits) != 6 or any(d < 0 or d > 9 for d in digits):
        raise ValueError("각 자리는 0~9 사이 숫자 6개여야 합니다")

    row = df.iloc[-1] if draw_no is None else df[df["draw_no"] == int(draw_no)].iloc[0]
    win_group = int(row["group_no"])
    win_digits = [int(row[c]) for c in DIGIT_COLS]

    k = _trailing_match(digits, win_digits)
    group_match = int(group) == win_group
    if k == 6 and group_match:
        tier, one_in, prize = "1등", 5_000_000, PRIZE_TIERS[0]["prize"]
    elif k == 6:
        tier, one_in, prize = "2등", 1_000_000, PRIZE_TIERS[1]["prize"]
    elif k >= 1:
        idx = {5: 2, 4: 3, 3: 4, 2: 5, 1: 6}[k]
        t = PRIZE_TIERS[idx]
        tier, one_in, prize = t["tier"], t["one_in"], t["prize"]
    else:
        tier, one_in, prize = "꽝", None, "미당첨"

    return CheckResult(tier, k, group_match, one_in, prize,
                       win_group, win_digits, int(row["draw_no"]))


def analysis_payload() -> dict:
    """Everything the web app's pension section needs, computed from clean data."""
    df = load_main()
    gf = group_frequency(df)
    pos = digit_position_frequency(df)
    sums = digit_sum_series(df)
    from scipy import stats
    # per-position chi-square uniformity of digits
    pos_p = []
    for pi in range(6):
        exp = np.full(10, pos[pi].sum() / 10)
        _, p = stats.chisquare(pos[pi], exp)
        pos_p.append(round(float(p), 4))
    # group uniformity chi-square
    _, gp = stats.chisquare(gf["count"].to_numpy(), np.full(5, gf["count"].sum() / 5))
    return {
        "n_draws": int(len(df)),
        "draw_min": int(df["draw_no"].min()),
        "draw_max": int(df["draw_no"].max()),
        "group_freq": gf["count"].astype(int).tolist(),
        "group_expected": round(float(gf["expected"].iloc[0]), 1),
        "group_chi2_p": round(float(gp), 4),
        "digit_position_freq": pos.tolist(),
        "digit_position_p": pos_p,
        "sum_mean": round(float(sums.mean()), 2),
        "sum_theory_mean": 27.0,   # 6 digits × mean 4.5
        "sum_hist": np.histogram(sums, bins=range(0, 56, 3))[0].tolist(),
        "sum_hist_edges": list(range(0, 56, 3)),
        "prize_tiers": PRIZE_TIERS,
        "latest": latest_draws(8, df),
    }
