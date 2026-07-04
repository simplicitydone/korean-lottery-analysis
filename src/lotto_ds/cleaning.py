"""cleaning.py — turn the raw Excel->SQLite dump into a tidy, validated clean DB.

The raw ``lottery.db`` is a verbatim dump of a spreadsheet workbook. It carries three
distinct kinds of dirt, none of which is in the *values* we care about but all of which
would embarrass a downstream analysis:

1. **Phantom columns** — ``draw_results`` has 399 ``extra_0..extra_398`` columns left over
   from trailing spreadsheet cells. 61 hold scattered stray values; the rest are fully null.
2. **Mixed-type / encoding-corrupted prize columns** — ``prize_1st_total`` is TEXT with a
   mojibake currency suffix (e.g. ``"2002006800��"`` — a mangled "원"), while sibling
   count/per-person columns are a mix of INTEGER, REAL and NULL. Recent draws have prizes
   entirely missing (not yet backfilled) — that is legitimate missingness, kept as NULL.
3. **Orphan sheet-dump tables** — ``number_stats``, ``consecutive_stats``, ``sum_odd_even``,
   ``empty_sheet``, ``verification``, ``number_freq_dist``, ``sum_freq_dist``,
   ``prize_calc_logic`` are other spreadsheet tabs dumped with generic ``col_0..col_N``
   headers and no schema. We do not trust them; every statistic is re-derived from the raw
   draw numbers instead.

This module reads the raw DB (never mutating it) and emits a normalized clean DB with tidy
tables plus a machine-readable cleaning report.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import (
    CLEAN_DB,
    LOTTO_NUMBER_MAX,
    LOTTO_NUMBER_MIN,
    LOTTO_PICKS,
    RAW_DB,
)

log = logging.getLogger("lotto_ds.cleaning")

WIN_COLS = [f"win{i}" for i in range(1, LOTTO_PICKS + 1)]
PENSION_DIGIT_COLS = [f"n{i}" for i in range(1, 7)]

# Orphan spreadsheet-tab tables we deliberately discard (stats are re-derived from raw draws).
ORPHAN_TABLES = (
    "number_stats",
    "consecutive_stats",
    "sum_odd_even",
    "empty_sheet",
    "verification",
    "number_freq_dist",
    "sum_freq_dist",
    "prize_calc_logic",
)


@dataclass
class CleaningReport:
    """Machine-readable record of what the pipeline changed — surfaced in notebook 01."""

    raw_draw_results_columns: int = 0
    phantom_columns_dropped: int = 0
    phantom_columns_with_stray_data: int = 0
    orphan_tables_dropped: list[str] = field(default_factory=list)
    lotto_rows: int = 0
    lotto_draw_range: tuple[int, int] = (0, 0)
    lotto_duplicate_draws: int = 0
    lotto_sequence_gaps: list[int] = field(default_factory=list)
    lotto_out_of_range_cells: int = 0
    prizes_parsed: int = 0
    prizes_missing: int = 0
    pension_main_rows: int = 0
    pension_bonus_rows: int = 0
    pension_draws: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "raw_draw_results_columns": self.raw_draw_results_columns,
            "phantom_columns_dropped": self.phantom_columns_dropped,
            "phantom_columns_with_stray_data": self.phantom_columns_with_stray_data,
            "orphan_tables_dropped": list(self.orphan_tables_dropped),
            "lotto_rows": self.lotto_rows,
            "lotto_draw_range": list(self.lotto_draw_range),
            "lotto_duplicate_draws": self.lotto_duplicate_draws,
            "lotto_sequence_gaps": list(self.lotto_sequence_gaps),
            "lotto_out_of_range_cells": self.lotto_out_of_range_cells,
            "prizes_parsed": self.prizes_parsed,
            "prizes_missing": self.prizes_missing,
            "pension_main_rows": self.pension_main_rows,
            "pension_bonus_rows": self.pension_bonus_rows,
            "pension_draws": self.pension_draws,
            "notes": list(self.notes),
        }


# ─────────────────────────────── raw readers ────────────────────────────────

def _connect(path: Path) -> sqlite3.Connection:
    if not Path(path).exists():
        raise FileNotFoundError(f"database not found: {path}")
    return sqlite3.connect(str(path))


def read_raw_draw_results(raw_db: Path = RAW_DB) -> pd.DataFrame:
    """Read ``draw_results`` verbatim, phantom columns and all — for 'before' profiling."""
    with _connect(raw_db) as conn:
        return pd.read_sql("SELECT * FROM draw_results ORDER BY draw_no", conn)


def list_raw_tables(raw_db: Path = RAW_DB) -> list[str]:
    with _connect(raw_db) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    return [r[0] for r in rows]


# ─────────────────────────────── value parsers ──────────────────────────────

# Matches leading digits of a possibly currency/mojibake-suffixed amount, e.g. "2002006800��".
_AMOUNT_RE = re.compile(r"-?\d[\d,]*")


def parse_krw_amount(value) -> int | None:
    """Parse a Korean-won amount that may be TEXT with a (possibly corrupted) currency suffix.

    ``"2002006800��"`` -> ``2002006800`` ; ``"0�"`` -> ``0`` ; blank/None -> ``None``.
    Legitimately-missing prize data (recent, un-backfilled draws) stays ``None`` — we do not
    fabricate zeros for it.
    """
    if value is None:
        return None
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, float):
        return None if pd.isna(value) else int(value)
    text = str(value).strip()
    if not text:
        return None
    match = _AMOUNT_RE.search(text.replace(",", ""))
    if not match:
        return None
    try:
        return int(match.group())
    except ValueError:
        log.warning("could not parse KRW amount from %r", value)
        return None


def _to_nullable_int(value) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        log.warning("could not coerce to int: %r", value)
        return None


# ─────────────────────────────── cleaners ───────────────────────────────────

def clean_lotto(raw_db: Path = RAW_DB, report: CleaningReport | None = None):
    """Return (draws, draw_numbers_long, prizes) tidy DataFrames from raw ``draw_results``.

    - ``draws``: one row/draw with the 6 winning numbers + bonus, wide (analysis convenience).
    - ``draw_numbers_long``: tidy long form — (draw_no, draw_date, position, number).
    - ``prizes``: parsed prize table with nullable integer KRW amounts.
    """
    report = report or CleaningReport()
    raw = read_raw_draw_results(raw_db)
    report.raw_draw_results_columns = raw.shape[1]

    phantom = [c for c in raw.columns if c.startswith("extra_")]
    report.phantom_columns_dropped = len(phantom)
    report.phantom_columns_with_stray_data = sum(
        1 for c in phantom if raw[c].notna().any()
    )

    core_cols = ["draw_no", "draw_date"] + WIN_COLS + ["bonus"]
    draws = raw[core_cols].copy()

    for col in WIN_COLS + ["bonus"]:
        draws[col] = pd.to_numeric(draws[col], errors="coerce").astype("Int64")

    draws["draw_no"] = draws["draw_no"].astype(int)
    draws["draw_date"] = pd.to_datetime(draws["draw_date"], errors="coerce").dt.date.astype(str)
    draws = draws.sort_values("draw_no").reset_index(drop=True)

    # ---- integrity checks (recorded, not silently swallowed) ----
    report.lotto_rows = len(draws)
    report.lotto_draw_range = (int(draws["draw_no"].min()), int(draws["draw_no"].max()))
    report.lotto_duplicate_draws = int(draws["draw_no"].duplicated().sum())
    full_seq = set(range(report.lotto_draw_range[0], report.lotto_draw_range[1] + 1))
    report.lotto_sequence_gaps = sorted(full_seq - set(draws["draw_no"].tolist()))
    oor = 0
    for col in WIN_COLS:
        vals = draws[col]
        oor += int(((vals < LOTTO_NUMBER_MIN) | (vals > LOTTO_NUMBER_MAX)).sum())
    report.lotto_out_of_range_cells = oor
    if report.lotto_duplicate_draws or report.lotto_sequence_gaps or oor:
        log.warning(
            "lotto integrity: dups=%d gaps=%d oor=%d",
            report.lotto_duplicate_draws,
            len(report.lotto_sequence_gaps),
            oor,
        )

    # ---- tidy long form ----
    long_rows = draws.melt(
        id_vars=["draw_no", "draw_date"],
        value_vars=WIN_COLS,
        var_name="position",
        value_name="number",
    )
    long_rows["position"] = long_rows["position"].str.replace("win", "", regex=False).astype(int)
    draw_numbers_long = long_rows.sort_values(["draw_no", "position"]).reset_index(drop=True)

    # ---- prizes ----
    prizes = pd.DataFrame({"draw_no": draws["draw_no"]})
    prizes["prize_1st_total_krw"] = raw["prize_1st_total"].map(parse_krw_amount).astype("Int64")
    prizes["prize_1st_winners"] = raw["prize_1st_count"].map(_to_nullable_int).astype("Int64")
    prizes["prize_1st_per_person_krw"] = raw["prize_1st_per_person"].map(_to_nullable_int).astype("Int64")
    prizes["prize_2nd_winners"] = raw["prize_2nd_count"].map(_to_nullable_int).astype("Int64")
    prizes["prize_2nd_per_person_krw"] = raw["prize_2nd_per_person"].map(_to_nullable_int).astype("Int64")
    prizes["prize_3rd_winners"] = raw["prize_3rd_count"].map(_to_nullable_int).astype("Int64")
    prizes["prize_3rd_per_person_krw"] = raw["prize_3rd_per_person"].map(_to_nullable_int).astype("Int64")
    report.prizes_parsed = int(prizes["prize_1st_total_krw"].notna().sum())
    report.prizes_missing = int(prizes["prize_1st_total_krw"].isna().sum())

    return draws, draw_numbers_long, prizes


def clean_pension(raw_db: Path = RAW_DB, report: CleaningReport | None = None):
    """Split ``pension_results`` into tidy main-draw and bonus-draw tables.

    Raw rows mix the drawn number set with a separate 'bonus' set via ``is_bonus``. Analyses
    almost always want just the main draw, so we separate them explicitly rather than making
    every downstream query remember the flag.
    """
    report = report or CleaningReport()
    with _connect(raw_db) as conn:
        raw = pd.read_sql("SELECT * FROM pension_results ORDER BY draw_no, is_bonus", conn)

    for col in ["draw_no", "group_no", "is_bonus", *PENSION_DIGIT_COLS]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").astype("Int64")
    raw["draw_date"] = pd.to_datetime(raw["draw_date"], errors="coerce").dt.date.astype(str)

    main = raw[raw["is_bonus"] == 0].drop(columns=["is_bonus"]).reset_index(drop=True)
    bonus = raw[raw["is_bonus"] == 1].drop(columns=["is_bonus", "group_no"]).reset_index(drop=True)

    report.pension_main_rows = len(main)
    report.pension_bonus_rows = len(bonus)
    report.pension_draws = int(main["draw_no"].nunique())
    return main, bonus


# ─────────────────────────────── writer / pipeline ──────────────────────────

def build_clean_db(raw_db: Path = RAW_DB, clean_db: Path = CLEAN_DB) -> CleaningReport:
    """Full pipeline: read raw, clean, validate, write normalized clean DB. Returns the report."""
    report = CleaningReport()
    report.orphan_tables_dropped = [
        t for t in list_raw_tables(raw_db) if t in ORPHAN_TABLES
    ]

    draws, draw_numbers_long, prizes = clean_lotto(raw_db, report)
    pension_main, pension_bonus = clean_pension(raw_db, report)

    clean_db = Path(clean_db)
    clean_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(clean_db)) as conn:
        draws.to_sql("draws", conn, if_exists="replace", index=False)
        draw_numbers_long.to_sql("draw_numbers", conn, if_exists="replace", index=False)
        prizes.to_sql("prizes", conn, if_exists="replace", index=False)
        pension_main.to_sql("pension_draws", conn, if_exists="replace", index=False)
        pension_bonus.to_sql("pension_bonus", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_draw_numbers_no ON draw_numbers(draw_no)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_draw_numbers_num ON draw_numbers(number)")
        conn.commit()

    report.notes.append(f"clean DB written to {clean_db}")
    log.info(
        "clean DB built: %d lotto draws, %d pension draws, %d phantom cols dropped",
        report.lotto_rows,
        report.pension_draws,
        report.phantom_columns_dropped,
    )
    return report


def load_clean(table: str, clean_db: Path = CLEAN_DB) -> pd.DataFrame:
    """Convenience loader for notebooks/web app: read a clean table as a DataFrame."""
    with _connect(clean_db) as conn:
        return pd.read_sql(f"SELECT * FROM {table}", conn)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rep = build_clean_db()
    import json

    print(json.dumps(rep.as_dict(), indent=2, ensure_ascii=False))
