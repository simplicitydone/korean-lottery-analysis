"""Tests for pension 720+ prize rules and the ticket checker."""

import pandas as pd
import pytest

from src.lotto_ds import pension as pn


def _fake_draw(group, digits):
    return pd.DataFrame([{
        "draw_no": 1, "draw_date": "2020-01-01", "group_no": group,
        **{f"n{i+1}": digits[i] for i in range(6)},
    }])


def test_trailing_match():
    assert pn._trailing_match([1, 2, 3, 4, 5, 6], [9, 9, 9, 4, 5, 6]) == 3
    assert pn._trailing_match([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]) == 6
    assert pn._trailing_match([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 0]) == 0


@pytest.mark.parametrize("group,digits,win_g,win_d,tier,k", [
    (2, [1, 2, 3, 4, 5, 6], 2, [1, 2, 3, 4, 5, 6], "1등", 6),   # group + all 6
    (3, [1, 2, 3, 4, 5, 6], 2, [1, 2, 3, 4, 5, 6], "2등", 6),   # all 6, wrong group
    (1, [9, 0, 2, 3, 4, 5], 2, [0, 0, 2, 3, 4, 5], "3등", 5),   # last 5 (only 1st digit differs)
    (1, [9, 9, 9, 3, 4, 5], 2, [0, 0, 0, 0, 4, 5], "6등", 2),   # last 2
    (1, [9, 9, 9, 9, 9, 5], 2, [0, 0, 0, 0, 0, 5], "7등", 1),   # last 1
    (1, [9, 9, 9, 9, 9, 9], 2, [0, 0, 0, 0, 0, 5], "꽝", 0),    # no match
])
def test_prize_tiers(group, digits, win_g, win_d, tier, k):
    df = _fake_draw(win_g, win_d)
    r = pn.check_ticket(group, digits, df=df)
    assert r.tier == tier
    assert r.matched_trailing == k


def test_check_validation():
    df = _fake_draw(1, [0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        pn.check_ticket(6, [0, 0, 0, 0, 0, 0], df=df)   # group out of range
    with pytest.raises(ValueError):
        pn.check_ticket(1, [0, 0, 0, 0, 0], df=df)       # too few digits
    with pytest.raises(ValueError):
        pn.check_ticket(1, [0, 0, 0, 0, 0, 10], df=df)   # digit out of range


def test_analysis_payload_shape():
    a = pn.analysis_payload()
    assert len(a["group_freq"]) == 5
    assert len(a["digit_position_freq"]) == 6
    assert all(len(row) == 10 for row in a["digit_position_freq"])
    assert len(a["prize_tiers"]) == 7
    assert a["n_draws"] > 0
