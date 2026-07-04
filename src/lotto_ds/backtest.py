"""backtest.py — leakage-free walk-forward evaluation of 'prediction' strategies.

The claim under test: *some* number-picking strategy beats blind random guessing. We evaluate
each strategy honestly — at draw *t* a strategy may use only draws `< t` — and score it by how
many of its 6 picks matched the 6 winning numbers.

The benchmark is not an opinion; it is arithmetic. Picking 6 distinct numbers from 45 when 6 are
winners is a hypergeometric draw, so the expected matches per ticket is exactly

    E[hits] = 6 * (6 / 45) = 0.8      (independent of any strategy)

A strategy "works" only if its mean hits sits *above* 0.8 by more than sampling error. None do.

Two comparison levels, because the legacy app generated **5 tickets per method** and reported the
best — and max-of-5 is mechanically larger than one ticket. We therefore compare per-ticket means
against 0.8, and best-of-5 means against a *random* best-of-5 baseline (never against 0.8).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import LOTTO_NUMBER_MAX, LOTTO_PICKS, RAW_DB
from .cleaning import WIN_COLS, load_clean

ALL_NUMBERS = np.arange(1, LOTTO_NUMBER_MAX + 1)
THEORETICAL_HITS = LOTTO_PICKS * (LOTTO_PICKS / LOTTO_NUMBER_MAX)  # 0.8


# ─────────────────────────── strategies ─────────────────────────────────────
# Each strategy maps prior-draw history -> a probability weight over numbers 1..45.
# We then sample 6 distinct numbers with those weights. Uniform weights == blind random.

def _weights_uniform(prior: np.ndarray) -> np.ndarray:
    return np.ones(LOTTO_NUMBER_MAX)


def _counts(prior: np.ndarray) -> np.ndarray:
    counts = np.zeros(LOTTO_NUMBER_MAX)
    if prior.size:
        vals, cnts = np.unique(prior.ravel(), return_counts=True)
        for v, c in zip(vals, cnts):
            if 1 <= v <= LOTTO_NUMBER_MAX:
                counts[int(v) - 1] = c
    return counts


def _weights_frequency(prior: np.ndarray) -> np.ndarray:
    return _counts(prior) + 1.0  # all-time hot (Laplace-smoothed)


def _weights_contrarian(prior: np.ndarray) -> np.ndarray:
    return 1.0 / (_counts(prior) + 1.0)  # least-frequent numbers favored


def _weights_hot_recent(prior: np.ndarray, window: int = 50) -> np.ndarray:
    return _counts(prior[-window:]) + 1.0


def _weights_cold(prior: np.ndarray) -> np.ndarray:
    # draws since last appearance -> favor long-absent numbers
    gap = np.full(LOTTO_NUMBER_MAX, prior.shape[0] if prior.ndim == 2 else 0, dtype=float)
    if prior.size:
        for i, draw in enumerate(prior):
            for v in draw:
                if 1 <= v <= LOTTO_NUMBER_MAX:
                    gap[int(v) - 1] = prior.shape[0] - i
    return gap + 1.0


STRATEGIES = {
    "random (uniform)": _weights_uniform,
    "frequency (all-time hot)": _weights_frequency,
    "hot (recent 50)": _weights_hot_recent,
    "cold (overdue)": _weights_cold,
    "contrarian (rare)": _weights_contrarian,
}


def _sample_ticket(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p = weights / weights.sum()
    return rng.choice(ALL_NUMBERS, size=LOTTO_PICKS, replace=False, p=p)


@dataclass
class BacktestResult:
    strategy: str
    n_draws: int
    tickets_per_draw: int
    mean_hits_per_ticket: float
    ci95_ticket: tuple[float, float]
    mean_best_of_k: float
    ci95_best: tuple[float, float]

    def as_row(self) -> dict:
        return {
            "strategy": self.strategy,
            "mean_hits/ticket": round(self.mean_hits_per_ticket, 4),
            "ticket 95% CI": f"[{self.ci95_ticket[0]:.3f}, {self.ci95_ticket[1]:.3f}]",
            f"mean best-of-{self.tickets_per_draw}": round(self.mean_best_of_k, 4),
            "beats 0.8?": "yes" if self.ci95_ticket[0] > THEORETICAL_HITS else "no",
        }


def _bootstrap_ci(x: np.ndarray, rng: np.random.Generator, n: int = 2000) -> tuple[float, float]:
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_backtest(
    strategy: str,
    draws: pd.DataFrame | None = None,
    start_draw: int = 300,
    tickets_per_draw: int = 5,
    seed: int = 42,
) -> BacktestResult:
    """Walk-forward evaluate one strategy. At draw t, only draws < t are visible."""
    if draws is None:
        draws = load_clean("draws")
    draws = draws.sort_values("draw_no").reset_index(drop=True)
    mat = draws[WIN_COLS].to_numpy(dtype=int)
    weight_fn = STRATEGIES[strategy]
    rng = np.random.default_rng(seed)

    ticket_hits, best_hits = [], []
    start_idx = int((draws["draw_no"] < start_draw).sum())
    for t in range(start_idx, len(mat)):
        prior = mat[:t]                      # strictly prior — no leakage
        actual = set(mat[t].tolist())
        w = weight_fn(prior)
        draw_hits = []
        for _ in range(tickets_per_draw):
            ticket = _sample_ticket(w, rng)
            draw_hits.append(len(actual & set(ticket.tolist())))
        ticket_hits.extend(draw_hits)
        best_hits.append(max(draw_hits))

    ticket_hits = np.array(ticket_hits, dtype=float)
    best_hits = np.array(best_hits, dtype=float)
    return BacktestResult(
        strategy=strategy,
        n_draws=len(best_hits),
        tickets_per_draw=tickets_per_draw,
        mean_hits_per_ticket=float(ticket_hits.mean()),
        ci95_ticket=_bootstrap_ci(ticket_hits, rng),
        mean_best_of_k=float(best_hits.mean()),
        ci95_best=_bootstrap_ci(best_hits, rng),
    )


def run_all_strategies(start_draw: int = 300, tickets_per_draw: int = 5, seed: int = 42) -> pd.DataFrame:
    draws = load_clean("draws")
    rows = [
        run_backtest(name, draws, start_draw, tickets_per_draw, seed).as_row()
        for name in STRATEGIES
    ]
    return pd.DataFrame(rows)


def load_legacy_archive(raw_db=RAW_DB) -> pd.DataFrame:
    """Load the legacy app's own leakage-free retrospective record for reconciliation.

    Returns per-method per-ticket hits so we can confirm the ML methods also mean ~0.8.
    """
    import json
    import sqlite3

    with sqlite3.connect(str(raw_db)) as conn:
        raw = pd.read_sql(
            "SELECT draw_no, method_results FROM prediction_accuracy_v3 "
            "WHERE method_results IS NOT NULL ORDER BY draw_no",
            conn,
        )
    records = []
    for _, row in raw.iterrows():
        methods = json.loads(row["method_results"])
        for name, payload in methods.items():
            # older rows use "hits"; recent lotto rows were backfilled with "hits_1st"
            for h in payload.get("hits") or payload.get("hits_1st") or []:
                records.append({"method": name.split(". ", 1)[-1], "hits": h})
    return pd.DataFrame(records)
