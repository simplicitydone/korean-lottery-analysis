"""Guardrail tests for the cleaning pipeline — the clean DB must stay trustworthy."""

import sqlite3

import pandas as pd
import pytest

from src.lotto_ds import LOTTO_NUMBER_MAX, LOTTO_NUMBER_MIN, RAW_DB
from src.lotto_ds import cleaning


@pytest.fixture(scope="module")
def clean_db(tmp_path_factory):
    out = tmp_path_factory.mktemp("clean") / "lotto_clean.db"
    report = cleaning.build_clean_db(raw_db=RAW_DB, clean_db=out)
    return out, report


def _read(db, table):
    with sqlite3.connect(str(db)) as conn:
        return pd.read_sql(f"SELECT * FROM {table}", conn)


def test_phantom_columns_dropped(clean_db):
    db, _ = clean_db
    draws = _read(db, "draws")
    assert not [c for c in draws.columns if c.startswith("extra_")]
    assert list(draws.columns) == [
        "draw_no", "draw_date", "win1", "win2", "win3", "win4", "win5", "win6", "bonus",
    ]


def test_orphan_tables_absent(clean_db):
    db, _ = clean_db
    with sqlite3.connect(str(db)) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    assert tables == {"draws", "draw_numbers", "prizes", "pension_draws", "pension_bonus"}
    assert not (tables & set(cleaning.ORPHAN_TABLES))


def test_lotto_integrity(clean_db):
    db, report = clean_db
    draws = _read(db, "draws")
    assert report.lotto_duplicate_draws == 0
    assert report.lotto_sequence_gaps == []
    assert report.lotto_out_of_range_cells == 0
    for col in cleaning.WIN_COLS:
        assert draws[col].between(LOTTO_NUMBER_MIN, LOTTO_NUMBER_MAX).all()
    # six distinct winning numbers per draw
    dupes = draws[cleaning.WIN_COLS].apply(lambda r: r.nunique() != 6, axis=1)
    assert not dupes.any()


def test_long_form_is_tidy(clean_db):
    db, _ = clean_db
    draws = _read(db, "draws")
    long = _read(db, "draw_numbers")
    assert len(long) == len(draws) * 6
    assert set(long["position"].unique()) == {1, 2, 3, 4, 5, 6}
    assert long["number"].between(LOTTO_NUMBER_MIN, LOTTO_NUMBER_MAX).all()


def test_prizes_parsed_to_int(clean_db):
    db, report = clean_db
    prizes = _read(db, "prizes")
    # mojibake currency suffix stripped -> plain integers, missing stays null (no fake zeros)
    parsed = prizes["prize_1st_total_krw"].dropna()
    assert (parsed >= 0).all()
    assert report.prizes_parsed + report.prizes_missing == len(prizes)
    assert report.prizes_missing >= 0


def test_pension_split(clean_db):
    db, report = clean_db
    main = _read(db, "pension_draws")
    assert "is_bonus" not in main.columns
    assert report.pension_draws == main["draw_no"].nunique()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2002006800��", 2002006800),
        ("0�", 0),
        ("1,234,567원", 1234567),
        ("", None),
        (None, None),
        (2500000000, 2500000000),
    ],
)
def test_parse_krw_amount(raw, expected):
    assert cleaning.parse_krw_amount(raw) == expected
