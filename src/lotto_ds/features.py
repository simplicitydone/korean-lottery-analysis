"""features.py — draw-level feature engineering on the clean lotto data.

Every feature is derived *fresh* from the winning numbers (the orphan spreadsheet stat tables
are ignored). These are the descriptive quantities the EDA and hypothesis-testing notebooks
lean on: per-draw shape (sum, odd/even, high/low, decade spread, AC value, consecutive runs,
max gap) and per-number history (frequency, gaps since last appearance).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from . import LOTTO_NUMBER_MAX, LOTTO_NUMBER_MIN, LOTTO_PICKS
from .cleaning import WIN_COLS, load_clean

ALL_NUMBERS = list(range(LOTTO_NUMBER_MIN, LOTTO_NUMBER_MAX + 1))


def _num_matrix(draws: pd.DataFrame) -> np.ndarray:
    """(n_draws, 6) sorted integer matrix of winning numbers."""
    m = draws[WIN_COLS].to_numpy(dtype=int)
    return np.sort(m, axis=1)


def ac_value(nums: list[int]) -> int:
    """Arithmetic Complexity: #distinct pairwise abs-differences minus (k-1).

    A lottery-community 'spread' heuristic. Max for a 6-pick is 10. Higher = more spread out.
    """
    diffs = {abs(a - b) for a, b in combinations(nums, 2)}
    return len(diffs) - (LOTTO_PICKS - 1)


def draw_features(draws: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return one row per draw with engineered shape features."""
    if draws is None:
        draws = load_clean("draws")
    m = _num_matrix(draws)

    out = pd.DataFrame({
        "draw_no": draws["draw_no"].to_numpy(),
        "draw_date": pd.to_datetime(draws["draw_date"]),
    })
    out["sum"] = m.sum(axis=1)
    out["mean"] = m.mean(axis=1)
    out["odd_count"] = (m % 2 != 0).sum(axis=1)
    out["even_count"] = LOTTO_PICKS - out["odd_count"]
    out["low_count"] = (m <= 22).sum(axis=1)          # 1..22 low, 23..45 high
    out["high_count"] = LOTTO_PICKS - out["low_count"]
    out["range"] = m[:, -1] - m[:, 0]
    out["max_gap"] = np.diff(m, axis=1).max(axis=1)
    out["consecutive_pairs"] = (np.diff(m, axis=1) == 1).sum(axis=1)
    out["ac_value"] = [ac_value(list(row)) for row in m]
    # decade spread: how many of the 5 decade-buckets (1-9,10-19,...,40-45) are represented
    decades = np.clip((m - 1) // 10, 0, 4)
    out["decades_covered"] = [len(set(row)) for row in decades]
    return out


def number_frequency(draws: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-number appearance count and share across all draws (main numbers only)."""
    if draws is None:
        draws = load_clean("draws")
    flat = draws[WIN_COLS].to_numpy(dtype=int).ravel()
    counts = pd.Series(flat).value_counts().reindex(ALL_NUMBERS, fill_value=0).sort_index()
    n_draws = len(draws)
    freq = pd.DataFrame({
        "number": counts.index,
        "count": counts.to_numpy(),
    })
    freq["share"] = freq["count"] / (n_draws * LOTTO_PICKS)
    # expected count under a uniform model: each draw picks 6 of 45 -> p = 6/45 per draw
    freq["expected_count"] = n_draws * LOTTO_PICKS / LOTTO_NUMBER_MAX
    return freq


def gaps_since_last(draws: pd.DataFrame | None = None) -> pd.Series:
    """Draws elapsed since each number last appeared (as of the most recent draw)."""
    if draws is None:
        draws = load_clean("draws")
    draws = draws.sort_values("draw_no")
    max_no = int(draws["draw_no"].max())
    mask_cols = draws[WIN_COLS].to_numpy(dtype=int)
    last_seen = {n: 0 for n in ALL_NUMBERS}
    for draw_no, row in zip(draws["draw_no"].to_numpy(), mask_cols):
        for n in row:
            last_seen[int(n)] = int(draw_no)
    return pd.Series({n: max_no - last_seen[n] for n in ALL_NUMBERS}, name="gap")


def sum_distribution_reference() -> dict:
    """Theoretical reference for the 6-of-45 sum: exact mean and the observed feasible range.

    The sum of 6 distinct picks from 1..45 ranges 21 (1..6) to 255 (40..45); by symmetry the
    exact mean is 6 * (1+45)/2 = 138.
    """
    return {"min": 21, "max": 255, "mean": LOTTO_PICKS * (LOTTO_NUMBER_MIN + LOTTO_NUMBER_MAX) / 2}


# ─────────────────────────── ML feature panel ───────────────────────────────

PANEL_FEATURES = ["freq_all", "freq_w10", "freq_w20", "freq_w50", "gap", "in_last", "in_last3"]


def _appearance_matrix(draws: pd.DataFrame) -> np.ndarray:
    """(n_draws, 45) binary matrix: 1 if number k appeared in draw t. Ordered by draw_no."""
    draws = draws.sort_values("draw_no")
    mat = np.zeros((len(draws), LOTTO_NUMBER_MAX), dtype=int)
    for i, row in enumerate(draws[WIN_COLS].to_numpy(dtype=int)):
        for n in row:
            if 1 <= n <= LOTTO_NUMBER_MAX:
                mat[i, n - 1] = 1
    return mat


def number_panel(draws: pd.DataFrame | None = None, min_history: int = 50) -> pd.DataFrame:
    """Build a leakage-free (draw × number) panel for supervised learning.

    The prediction task, framed honestly as binary classification:
        *"will number k appear in the NEXT draw?"*
    For each draw t (with ≥ `min_history` prior draws) and each number k=1..45, the features use only
    information available **through draw t**, and the label is whether k appears in draw t+1. There
    is therefore no look-ahead leakage anywhere in the panel.

    Features per (t, k):
      freq_all  cumulative appearance rate through t
      freq_w10/20/50  appearance count in the trailing 10/20/50 draws
      gap       draws since k last appeared (0 = appeared in draw t)
      in_last   appeared in draw t
      in_last3  appeared in any of the last 3 draws
    Label: appeared_next (0/1).
    """
    if draws is None:
        draws = load_clean("draws")
    draws = draws.sort_values("draw_no").reset_index(drop=True)
    appear = _appearance_matrix(draws)
    n = len(appear)
    draw_nos = draws["draw_no"].to_numpy()

    cum = np.cumsum(appear, axis=0)                        # cum[t,k] = appearances through t

    def window_count(w):
        pad = np.zeros((w, LOTTO_NUMBER_MAX), dtype=int)
        padded = np.vstack([pad, cum])
        return cum - padded[:n]                            # cum[t] - cum[t-w]

    w10, w20, w50 = window_count(10), window_count(20), window_count(50)

    # gap[t,k]: draws since last appearance (as of t). Vectorized running update.
    gap = np.zeros((n, LOTTO_NUMBER_MAX), dtype=int)
    last = np.full(LOTTO_NUMBER_MAX, -1)
    for t in range(n):
        gap[t] = np.where(last < 0, t + 1, t - last)
        last = np.where(appear[t] == 1, t, last)

    in_last3 = np.zeros((n, LOTTO_NUMBER_MAX), dtype=int)
    for t in range(n):
        lo = max(0, t - 2)
        in_last3[t] = (appear[lo:t + 1].sum(axis=0) > 0).astype(int)

    rows = []
    for t in range(min_history, n - 1):                    # need t+1 for the label
        label = appear[t + 1]
        for k in range(LOTTO_NUMBER_MAX):
            rows.append((
                int(draw_nos[t]), k + 1,
                cum[t, k] / (t + 1), w10[t, k], w20[t, k], w50[t, k],
                gap[t, k], appear[t, k], in_last3[t, k],
                int(label[k]),
            ))
    cols = ["draw_no", "number", *PANEL_FEATURES, "appeared_next"]
    return pd.DataFrame(rows, columns=cols)
