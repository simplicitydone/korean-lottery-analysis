"""lotto_ds — an honest data-science study of the Korean 6/45 lottery and pension lottery.

Modules
-------
cleaning     : raw Excel->SQLite dump  ->  tidy, validated clean DB
features     : draw-level feature engineering (sum, odd/even, AC value, gaps, ...)
stats_tests  : inferential tests for uniformity / independence / randomness
backtest     : leakage-free walk-forward evaluation of "prediction" strategies

Design note
-----------
The raw ``lottery.db`` is never mutated (the legacy Flask app still reads it). The cleaning
pipeline reads it and writes a fresh normalized ``data/clean/lotto_clean.db`` that every
notebook and the reframed web app share as the single source of truth.
"""

from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_ROOT.parent.parent

RAW_DB = PROJECT_ROOT / "lottery.db"
CLEAN_DB = PROJECT_ROOT / "data" / "clean" / "lotto_clean.db"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

LOTTO_NUMBER_MIN = 1
LOTTO_NUMBER_MAX = 45
LOTTO_PICKS = 6

__all__ = [
    "PROJECT_ROOT",
    "RAW_DB",
    "CLEAN_DB",
    "FIGURES_DIR",
    "LOTTO_NUMBER_MIN",
    "LOTTO_NUMBER_MAX",
    "LOTTO_PICKS",
]
