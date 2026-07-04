"""evaluate.py — score a ticket by how well its *shape* matches historical winners.

This is the legacy app's "적합도 평가" reborn honestly. Given 6 numbers, it grades how *typical* the
combination looks compared to the distribution of past winning-number shapes (sum, odd/even split,
high/low balance, AC value, consecutive runs, decade spread). Each criterion's target band is derived
**from the data** (percentiles), not hardcoded.

The honest caveat, stated on every result: **a high score does not improve your odds.** Every 6/45
combination has exactly the same 1-in-8,145,060 chance. What the score really tells you is whether
your pick looks like the "crowd" — and picking a crowd-shaped combo means, *if* it ever won, you'd
split the prize with more people. So a low "typicality" score can actually be financially smarter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import LOTTO_NUMBER_MAX, LOTTO_PICKS
from .features import ac_value, draw_features
from .pension import DIGIT_COLS, GROUPS, load_main

HONEST_NOTE = ("이 점수는 과거 당첨 조합의 '모양'과 얼마나 비슷한지를 나타낼 뿐, "
               "당첨 확률과는 무관합니다. 모든 조합의 확률은 동일합니다. "
               "오히려 흔한 모양(높은 점수)은 당첨 시 더 많은 사람과 상금을 나눠야 할 수 있습니다.")

GRADES = [(85, "S"), (70, "A"), (55, "B"), (40, "C"), (0, "F")]


def _grade(score: float) -> str:
    for thr, g in GRADES:
        if score >= thr:
            return g
    return "F"


def evaluate_lotto(numbers, draws: pd.DataFrame | None = None) -> dict:
    nums = sorted(int(n) for n in numbers)
    if len(nums) != 6 or len(set(nums)) != 6 or any(n < 1 or n > LOTTO_NUMBER_MAX for n in nums):
        raise ValueError("서로 다른 1~45 번호 6개를 입력하세요.")

    feat = draw_features(draws)
    s = sum(nums)
    odd = sum(1 for n in nums if n % 2)
    high = sum(1 for n in nums if n >= 23)
    ac = ac_value(nums)
    consec = sum(1 for i in range(5) if nums[i + 1] - nums[i] == 1)
    decades = len({min((n - 1) // 10, 4) for n in nums})

    def band(series, lo=10, hi=90):
        return float(np.percentile(series, lo)), float(np.percentile(series, hi))

    # each criterion: full points if in the "ideal" (p25–p75) band, half if in the "typical"
    # (p10–p90) band, else 0. Categorical ones use the historically common values.
    s_lo, s_hi = band(feat["sum"], 25, 75)
    s_tlo, s_thi = band(feat["sum"], 10, 90)
    ac_med = float(np.percentile(feat["ac_value"], 40))

    criteria = []

    def add(name, value, target, full, points):
        got = full if points == "full" else (round(full / 2) if points == "half" else 0)
        criteria.append({"name": name, "value": value, "target": target,
                         "got": got, "max": full,
                         "pass": points == "full", "partial": points == "half"})

    add("합계 (Sum)", s, f"{s_lo:.0f}–{s_hi:.0f}", 25,
        "full" if s_lo <= s <= s_hi else ("half" if s_tlo <= s <= s_thi else "none"))
    add("홀짝 (Odd:Even)", f"{odd}:{6-odd}", "2:4 ~ 4:2", 20,
        "full" if odd in (2, 3, 4) else ("half" if odd in (1, 5) else "none"))
    add("고저 (High:Low)", f"{high}:{6-high}", "2:4 ~ 4:2", 15,
        "full" if high in (2, 3, 4) else ("half" if high in (1, 5) else "none"))
    add("AC 값 (복잡도)", ac, f"≥ {ac_med:.0f}", 20,
        "full" if ac >= ac_med else ("half" if ac >= ac_med - 2 else "none"))
    add("연속번호 (Consecutive)", consec, "0–1쌍", 10,
        "full" if consec <= 1 else ("half" if consec == 2 else "none"))
    add("십분위 분포 (Spread)", f"{decades}구간", "≥ 3구간", 10,
        "full" if decades >= 3 else ("half" if decades == 2 else "none"))

    score = sum(c["got"] for c in criteria)
    return {"mode": "LOTTO", "numbers": nums, "score": score, "grade": _grade(score),
            "criteria": criteria, "note": HONEST_NOTE}


def evaluate_pension(group, digits, df: pd.DataFrame | None = None) -> dict:
    g = int(group)
    ds = [int(d) for d in digits]
    if not (1 <= g <= 5) or len(ds) != 6 or any(d < 0 or d > 9 for d in ds):
        raise ValueError("조는 1~5, 각 자리는 0~9 숫자 6개여야 합니다.")

    if df is None:
        df = load_main()
    sums = df[DIGIT_COLS].to_numpy(float).sum(axis=1)
    s = sum(ds)
    s_lo, s_hi = float(np.percentile(sums, 25)), float(np.percentile(sums, 75))
    unique = len(set(ds))
    # group frequency percentile (purely descriptive — groups are uniform)
    gcounts = df["group_no"].value_counts().reindex(GROUPS, fill_value=0)
    g_rank = float((gcounts <= gcounts.get(g, 0)).mean() * 100)

    criteria = []
    criteria.append({"name": "자릿수 합 (Digit sum)", "value": s, "target": f"{s_lo:.0f}–{s_hi:.0f}",
                     "got": 50 if s_lo <= s <= s_hi else 25, "max": 50,
                     "pass": s_lo <= s <= s_hi, "partial": not (s_lo <= s <= s_hi)})
    criteria.append({"name": "자릿수 다양성 (Variety)", "value": f"{unique}종", "target": "≥ 4종",
                     "got": 30 if unique >= 4 else 15, "max": 30,
                     "pass": unique >= 4, "partial": unique < 4})
    criteria.append({"name": f"조 빈도 (Group {g})", "value": f"{g_rank:.0f}%ile",
                     "target": "참고용", "got": 20, "max": 20, "pass": True, "partial": False})
    score = sum(c["got"] for c in criteria)
    note = HONEST_NOTE + " 연금복권은 각 자리가 완전 균등·독립이라 '모양'이 사실상 무의미합니다."
    return {"mode": "PENSION", "group": g, "digits": ds, "score": score,
            "grade": _grade(score), "criteria": criteria, "note": note}
