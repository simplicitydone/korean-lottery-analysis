"""Tests for the randomness battery — it must pass fair data and BITE a rigged stream."""

import numpy as np
import pandas as pd

from src.lotto_ds import randomness as rd

WIN = [f"win{i}" for i in range(1, 7)]


def _fair(n=800, seed=0):
    rng = np.random.default_rng(seed)
    rows = [np.sort(rng.choice(range(1, 46), size=6, replace=False)) for _ in range(n)]
    return pd.DataFrame(rows, columns=WIN)


def _rigged(n=800, seed=0):
    """Number 1 is forced into every draw — a blatant non-uniformity."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        rest = rng.choice(range(2, 46), size=5, replace=False)
        rows.append(np.sort(np.append(rest, 1)))
    return pd.DataFrame(rows, columns=WIN)


def test_battery_shape():
    b = rd.run_battery(_fair())
    assert len(b) == 6
    assert {"test", "statistic", "p_value", "random_consistent"} <= set(b.columns)


def test_fair_data_mostly_passes():
    b = rd.run_battery(_fair(seed=5))
    # entropy and KS-drift are the robust core checks; they must pass on fair data
    assert bool(b.set_index("test").filter(like="엔트로피", axis=0)["random_consistent"].iloc[0])
    # overall the majority of the battery is random-consistent
    assert b["random_consistent"].sum() >= 4


def test_rigged_stream_is_flagged_by_entropy():
    # Forcing number 1 every draw destroys uniformity -> entropy test must reject.
    r = rd.shannon_entropy(_rigged(seed=2))
    assert r.reject_null  # p < alpha => not uniform
    assert r.statistic < 1.0  # entropy efficiency below the uniform maximum


def test_rigged_flagged_in_battery():
    b = rd.run_battery(_rigged(seed=1))
    entropy_row = b[b["test"].str.contains("엔트로피")].iloc[0]
    assert not entropy_row["random_consistent"]
