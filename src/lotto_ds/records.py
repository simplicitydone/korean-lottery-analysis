"""records.py — the per-draw prediction accuracy archive ("actual records like before").

The legacy app ran a leakage-free retrospective engine that, for each past draw, generated picks
from several methods *using only prior data* and recorded how many matched. Those records live in
`prediction_accuracy_v3` (lotto, 500 draws) and `prediction_accuracy_pension_v3` (pension). This
module reads them and exposes per-회차 rows, a per-draw method breakdown, and a method leaderboard —
so the honest verdict (§09) is backed by browsable, per-draw evidence rather than a single number.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from . import RAW_DB

_LOTTO_TABLE = "prediction_accuracy_v3"
_PENSION_TABLE = "prediction_accuracy_pension_v3"


def _connect():
    return sqlite3.connect(str(RAW_DB))


def _loads(s):
    try:
        return json.loads(s) if s else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _method_hits(m: dict) -> list:
    """Per-set hit counts for a method, tolerating the archive's two schemas.

    Older lotto rows use ``hits``; recent lotto rows were backfilled with the pension-style
    ``hits_1st`` key. Pension rows use ``hits_1st`` (main-prize match count). Return whichever exists.
    """
    return m.get("hits") or m.get("hits_1st") or []


def lotto_records(limit: int = 60) -> list[dict]:
    """Recent per-draw records: actual numbers + best hit + per-method best hit."""
    with _connect() as conn:
        rows = pd.read_sql(
            f"SELECT draw_no, draw_date, actual_nums, ensemble_best, ensemble_hits, "
            f"method_results, training_size FROM {_LOTTO_TABLE} ORDER BY draw_no DESC LIMIT ?",
            conn, params=(limit,))
    out = []
    for _, r in rows.iterrows():
        methods = _loads(r["method_results"])
        per_method = {name.split(". ", 1)[-1]: int(max(_method_hits(m) or [0]))
                      for name, m in methods.items()}
        out.append({
            "draw_no": int(r["draw_no"]),
            "draw_date": r["draw_date"],
            "actual": _loads(r["actual_nums"]),
            "ensemble_best": int(r["ensemble_best"] or 0),
            "ensemble_hits": _loads(r["ensemble_hits"]),
            "method_best": per_method,
            "training_size": int(r["training_size"] or 0),
        })
    return out


def pension_records(limit: int = 40) -> list[dict]:
    with _connect() as conn:
        rows = pd.read_sql(
            f"SELECT draw_no, draw_date, actual_nums, ensemble_best, method_results, "
            f"training_size FROM {_PENSION_TABLE} ORDER BY draw_no DESC LIMIT ?",
            conn, params=(limit,))
    out = []
    for _, r in rows.iterrows():
        methods = _loads(r["method_results"])
        per_method = {name.split(". ", 1)[-1]: int(m.get("best_1st", 0) or 0)
                      for name, m in methods.items()}
        out.append({
            "draw_no": int(r["draw_no"]),
            "draw_date": r["draw_date"],
            "actual": _loads(r["actual_nums"]),
            "ensemble_best": int(r["ensemble_best"] or 0),
            "method_best": per_method,
            "training_size": int(r["training_size"] or 0),
        })
    return out


def draw_detail(draw_no: int, mode: str = "LOTTO") -> dict:
    """Full per-draw breakdown: every method's generated sets and their hits."""
    table = _PENSION_TABLE if mode.upper() == "PENSION" else _LOTTO_TABLE
    with _connect() as conn:
        rows = pd.read_sql(
            f"SELECT * FROM {table} WHERE draw_no=?", conn, params=(int(draw_no),))
    if rows.empty:
        return {}
    r = rows.iloc[0]
    return {
        "draw_no": int(r["draw_no"]),
        "draw_date": r["draw_date"],
        "actual": _loads(r["actual_nums"]),
        "methods": _loads(r["method_results"]),
        "training_size": int(r["training_size"] or 0),
    }


def method_leaderboard(mode: str = "LOTTO") -> list[dict]:
    """Per-method mean and best hits across the whole archive."""
    table = _PENSION_TABLE if mode.upper() == "PENSION" else _LOTTO_TABLE
    with _connect() as conn:
        rows = pd.read_sql(f"SELECT method_results FROM {table} "
                           f"WHERE method_results IS NOT NULL", conn)
    agg: dict[str, list[int]] = {}
    for _, r in rows.iterrows():
        for name, m in _loads(r["method_results"]).items():
            agg.setdefault(name.split(". ", 1)[-1], []).extend(_method_hits(m))
    board = [{"method": k, "mean_hits": round(float(np.mean(v)), 3),
              "best_hits": int(np.max(v)) if v else 0, "n_tickets": len(v)}
             for k, v in agg.items()]
    return sorted(board, key=lambda x: -x["mean_hits"])


def coverage(mode: str = "LOTTO") -> dict:
    table = _PENSION_TABLE if mode.upper() == "PENSION" else _LOTTO_TABLE
    with _connect() as conn:
        n, lo, hi = conn.execute(
            f"SELECT COUNT(*), MIN(draw_no), MAX(draw_no) FROM {table}").fetchone()
    return {"computed": int(n or 0), "min": int(lo or 0), "max": int(hi or 0)}
