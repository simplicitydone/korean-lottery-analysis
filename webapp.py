"""webapp.py — the reframed web app: a Data Science Report on the Korean lotteries.

This supersedes the legacy prediction SPA (`app.py` + `static/`). It reuses the `lotto_ds`
package (the same code the notebooks use) and serves the four-part story — cleaning, EDA,
hypothesis testing, prediction backtest — as a single JSON payload rendered by a self-contained
report page. No auth gate, no "predictor": the honest analysis is the product.

Run:  python webapp.py        (dev)   ·   waitress-serve --port=5001 webapp:app   (prod)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from flask import Flask, jsonify, send_from_directory
from statsmodels.tsa.stattools import acf

from src.lotto_ds import CLEAN_DB, viz
from src.lotto_ds import backtest as bt
from src.lotto_ds import bayesian as by
from src.lotto_ds import cleaning, features
from src.lotto_ds import ml_models as ml
from src.lotto_ds import probability as pb
from src.lotto_ds import randomness as rd
from src.lotto_ds import stats_tests as st
from src.lotto_ds import unsupervised as un

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("lotto_ds.webapp")

app = Flask(__name__, static_folder="static_report", static_url_path="")

_REPORT_CACHE: dict | None = None


def _to_native(obj):
    """Recursively convert numpy scalars/arrays to plain Python and NaN/inf to None.

    NaN and Infinity are not valid JSON — strict browser ``JSON.parse`` rejects them — so a NaN
    ``df`` (tests without degrees of freedom) must serialize as ``null``, not ``NaN``.
    """
    import math

    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _ensure_clean_db():
    if not CLEAN_DB.exists():
        log.info("clean DB missing — building it from raw lottery.db ...")
        cleaning.build_clean_db()


def build_report() -> dict:
    """Assemble the full report payload once; cheap enough to compute on first request."""
    _ensure_clean_db()
    draws = cleaning.load_clean("draws")
    feat = features.draw_features(draws)
    freq = features.number_frequency(draws)
    n_draws = len(draws)

    # --- cleaning summary (re-run pipeline in-memory for the live numbers) ---
    report = cleaning.CleaningReport()
    report.orphan_tables_dropped = [
        t for t in cleaning.list_raw_tables() if t in cleaning.ORPHAN_TABLES
    ]
    cleaning.clean_lotto(report=report)
    cleaning.clean_pension(report=report)
    clean_summary = report.as_dict()

    # --- EDA series ---
    expected = float(freq["expected_count"].iloc[0])
    sums = feat["sum"].to_numpy()
    hist_counts, hist_edges = np.histogram(sums, bins=range(40, 246, 6))
    ref = features.sum_distribution_reference()

    odd_dist = feat["odd_count"].value_counts().reindex(range(7), fill_value=0).sort_index()
    from scipy import stats as scistats
    binom_exp = (scistats.binom.pmf(range(7), 6, 23 / 45) * n_draws).round(1)

    roll = feat.set_index("draw_no")["sum"].rolling(50).mean()

    # --- hypothesis tests ---
    chi = st.chi_square_number_uniformity(draws)
    obs = pd.Series(draws[cleaning.WIN_COLS].to_numpy(int).ravel()).value_counts()
    obs = obs.reindex(range(1, 46), fill_value=0).sort_index()
    resid = ((obs - expected) / np.sqrt(expected)).round(3)
    acf_vals = acf(sums, nlags=15, fft=False)[1:]
    battery = st.run_all().to_dict(orient="records")

    # --- backtest ---
    backtest_rows = []
    for name in bt.STRATEGIES:
        r = bt.run_backtest(name, draws=draws, start_draw=300, tickets_per_draw=5)
        backtest_rows.append({
            "strategy": r.strategy,
            "mean": round(r.mean_hits_per_ticket, 4),
            "ci_lo": round(r.ci95_ticket[0], 4),
            "ci_hi": round(r.ci95_ticket[1], 4),
        })
    archive = bt.load_legacy_archive()
    method_stats = archive.groupby("method")["hits"].agg(["mean", "sem"]).reset_index()
    legacy_rows = [
        {
            "method": row["method"],
            "mean": round(float(row["mean"]), 4),
            "ci_lo": round(float(row["mean"] - 1.96 * row["sem"]), 4),
            "ci_hi": round(float(row["mean"] + 1.96 * row["sem"]), 4),
            "is_ml": row["method"] in {"Deep Learning", "Random Forest", "DL+ML Ensemble"},
        }
        for _, row in method_stats.sort_values("mean").iterrows()
    ]
    ml_bt = ml.topk_backtest()

    # --- probability (theory) ---
    tiers = pb.prize_tier_table()

    # --- bayesian ---
    post = by.posterior_table(draws, prior_strength=45).sort_values("posterior_mean")

    # --- randomness battery ---
    battery_rand = rd.run_battery(draws)

    # --- ML (honest supervised) ---
    ml_results = ml.evaluate_models()
    gbm = next(r for r in ml_results if "Gradient" in r.model_name)

    # --- unsupervised ---
    pca = un.pca_analysis(draws)
    km = un.kmeans_silhouettes(draws)
    tsne = un.tsne_embedding(draws)

    return {
        "meta": {
            "n_draws": n_draws,
            "draw_min": int(draws["draw_no"].min()),
            "draw_max": int(draws["draw_no"].max()),
            "date_min": str(draws["draw_date"].min()),
            "date_max": str(draws["draw_date"].max()),
            "palette": {"cat": viz.CAT, "seq": viz.SEQ, "status": viz.STATUS},
        },
        "cleaning": clean_summary,
        "eda": {
            "freq_numbers": list(range(1, 46)),
            "freq_counts": freq["count"].astype(int).tolist(),
            "freq_expected": round(expected, 1),
            "sum_hist_counts": hist_counts.tolist(),
            "sum_hist_edges": list(hist_edges),
            "sum_mean": round(float(sums.mean()), 2),
            "sum_theory_mean": ref["mean"],
            "sum_std": round(float(sums.std()), 2),
            "odd_dist": odd_dist.astype(int).tolist(),
            "odd_binom_expected": binom_exp.tolist(),
            "roll_draw_no": [int(x) for x in roll.dropna().index.tolist()],
            "roll_sum": [round(float(x), 2) for x in roll.dropna().tolist()],
        },
        "tests": {
            "battery": battery,
            "chi2": {"stat": round(chi.statistic, 2), "df": chi.df, "p": round(chi.p_value, 4)},
            "residuals": resid.tolist(),
            "acf": [round(float(x), 4) for x in acf_vals],
            "acf_conf": round(1.96 / np.sqrt(len(sums)), 4),
        },
        "backtest": {
            "theoretical": bt.THEORETICAL_HITS,
            "strategies": backtest_rows,
            "ml_strategy": {"mean": round(ml_bt["mean"], 4), "ci_lo": round(ml_bt["ci_lo"], 4),
                            "ci_hi": round(ml_bt["ci_hi"], 4)},
            "legacy_methods": legacy_rows,
            "legacy_ticket_count": int(len(archive)),
        },
        "probability": {
            "total_combos": pb.comb(45, 6),
            "expected_matches": pb.expected_matches(),
            "tiers": [{"k": t["matches"], "one_in": round(t["one_in"])} for t in tiers],
            "within_draw_cov": round(pb.within_draw_covariance(), 6),
            "sum_theory": {"mean": pb.sum_theory().mean, "std": round(pb.sum_theory().std, 2)},
        },
        "bayesian": {
            "fair_theta": round(by.FAIR_THETA, 4),
            "numbers": post["number"].astype(int).tolist(),
            "posterior_mean": [round(x, 4) for x in post["posterior_mean"]],
            "ci_lo": [round(x, 4) for x in post["ci_lo"]],
            "ci_hi": [round(x, 4) for x in post["ci_hi"]],
            "empirical_rate": [round(x, 4) for x in post["empirical_rate"]],
            "contains_fair": post["ci_contains_fair"].astype(bool).tolist(),
            "contains_fair_count": int(post["ci_contains_fair"].sum()),
        },
        "randomness": {
            "battery": battery_rand[["test", "statistic", "p_value", "random_consistent"]]
            .to_dict(orient="records"),
            "pass_count": int(battery_rand["random_consistent"].sum()),
            "total": int(len(battery_rand)),
        },
        "ml": {
            "chance_auc": 0.5,
            "base_rate": round(ml.BASE_RATE, 4),
            "models": [r.as_row() for r in ml_results],
            "roc": {r.model_name: r.roc for r in ml_results},
            "calibration": {r.model_name: r.calibration for r in ml_results},
            "importances": gbm.importances,
        },
        "unsupervised": {
            "pca_evr": [round(x, 4) for x in pca["explained_variance_ratio"]],
            "silhouettes": {str(k): round(v, 3) for k, v in km["silhouettes"].items()},
            "best_silhouette": round(km["best_score"], 3),
            "tsne": [[round(float(x), 2), round(float(y), 2)] for x, y in tsne],
            "tsne_color": feat["draw_no"].astype(int).tolist(),
        },
    }


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/report")
def report():
    global _REPORT_CACHE
    if _REPORT_CACHE is None:
        log.info("building report payload (first request) ...")
        _REPORT_CACHE = _to_native(build_report())
    return jsonify(_REPORT_CACHE)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "clean_db": CLEAN_DB.exists()})


if __name__ == "__main__":
    import os

    _ensure_clean_db()
    port = int(os.environ.get("PORT", "5001"))
    app.run(debug=os.environ.get("FLASK_DEBUG", "1") != "0", host="0.0.0.0", port=port)
