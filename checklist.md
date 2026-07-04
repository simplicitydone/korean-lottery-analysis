# Do-over Checklist — Korean Lottery: An Honest Data-Science Study

> 로또 6/45 + 연금복권 데이터를 이용한 **정직한 데이터 사이언스 스터디**.
> "예측 엔진"에서 "데이터 정제 → EDA → 가설검정 → 예측 반증(backtest)" 서사로 전환.
> Bilingual: Korean narrative, English code/technical terms.

## Phase 0 — Setup ✅
- [x] `.venv` + DS deps (pandas, numpy, scipy, statsmodels, sklearn, matplotlib, jupyter, nbformat, jupytext)
- [x] `requirements-ds.txt` pinned
- [x] `src/lotto_ds/` package skeleton (`__init__`, `cleaning`, `viz`)
- [x] `context-notes.md` seeded

## Phase 1 — Data cleaning (src/lotto_ds/cleaning.py + notebooks/01) ✅
- [x] Document the raw mess: 399 phantom `extra_*` cols, 8 orphan Excel-sheet tables, currency-TEXT prize cols
- [x] `read_raw_draw_results()` — read raw lottery.db as-is
- [x] `clean_lotto()` — drop phantom cols, parse currency→int prizes, validate ranges/dupes/gaps, tidy long-form numbers table
- [x] `clean_pension()` — split main vs bonus rows (is_bonus), validate 322 draws
- [x] Write `data/clean/lotto_clean.db` (draws, draw_numbers, prizes, pension_draws, pension_bonus)
- [x] Cleaning validation assertions + `tests/test_cleaning.py` (12 passing)
- [x] Notebook 01: before/after profiling, bilingual narrative — executes clean via nbconvert

## Phase 2 — EDA (features.py + notebooks/02) ✅
- [x] `features.py`: sum, odd/even, high/low, AC value, consecutive, gaps, decade buckets, freq
- [x] Frequency distribution + expected-uniform overlay
- [x] Sum distribution vs theoretical (CLT/normal approx) — obs mean 138.29 ≈ theory 138
- [x] Time-series: rolling sum stability (no trend)
- [x] Pension digit-position heatmap
- [x] Charts follow dataviz palette (`viz.py`); exported to reports/figures/; notebook executes clean

## Phase 3 — Hypothesis testing (stats_tests.py + notebooks/03) ✅
- [x] Chi-square goodness-of-fit: numbers uniform? → p=0.97, no (std-residual chart)
- [x] Runs test / Wald-Wolfowitz odd/even sequence → p=0.25, random
- [x] Autocorrelation of draw sums (Ljung-Box) → borderline p=0.032, honestly debunked via effect size + Bonferroni + lag-robustness
- [x] "Hot number" edge test (walk-forward) → p=0.42, no edge vs 6·6/45 baseline
- [x] Pension digit uniformity per position → all p>0.05
- [x] Bilingual verdicts + effect sizes; notebook 03 executes clean

## Phase 4 — Prediction backtest / debunk (backtest.py + notebooks/04) ✅
- [x] Walk-forward eval harness (train on prior draws only — no leakage)
- [x] 5 strategies: random, all-time frequency, hot, cold, contrarian — all ≈0.80, CIs straddle 0.8
- [x] Metric: expected hits vs theoretical 6·6/45=0.8, bootstrap CIs, best-of-5 handled honestly
- [x] Negative-EV prize-tier analysis (hypergeometric odds + expected ROI)
- [x] Reconciled with app's own `prediction_accuracy_v3` (19,560 tickets, 8 methods incl. DL/RF all ≈0.80)

## Phase 5 — Web app reframe ✅
- [x] New `webapp.py` — DS report API (`/api/report`) reusing lotto_ds modules from clean DB; no auth, no fake predict
- [x] New `static_report/` — self-contained bilingual "Data Science Report" SPA (Cleaning · EDA · Tests · Backtest), theme-aware, validated palette, Chart.js
- [x] Numpy/NaN→JSON sanitizer; endpoints smoke-tested (health/report/index all 200); rendered + visually verified via headless Edge
- [x] Legacy `app.py`/`static/` left intact for reference (surgical)

## Phase 6 — Docs & polish ✅
- [x] README rewritten as bilingual portfolio landing (methodology, findings, run steps, reproducibility)
- [x] requirements-ds.txt pinned (incl. flask/waitress); .gitignore ignores regenerable data/clean/
- [x] scratch/ left as-is (pre-existing one-offs; not mine to delete — noted in README)
- [x] Final verification: py_compile OK, 12 tests pass, clean DB rebuilds, all 4 notebooks execute clean, webapp boots

---

## v2 Refinement — Educational Showcase (2026-07-05)

Goal: portfolio "show-off" piece; no password; full teaching depth in notebooks AND web app;
keep pension as a probability contrast; add SOTA technique families.

- [x] **Phase A** — Deleted legacy prediction app (app.py, static/, engines, auto_updater,
      backfill, lotto_dashboard, scratch/, scrape HTML). Retargeted deploy: serve.py→webapp,
      Dockerfile/compose/.env stripped of secrets, requirements split (runtime vs -dev).
- [x] **Phase B** — New modules: `probability.py` (초기하·CLT·독립성), `randomness.py` (엔트로피·
      KS·AD·ADF·순열 스펙트럼), `bayesian.py` (Beta-Binomial), `ml_models.py` (walk-forward
      GBM/로지스틱·ROC·보정·중요도·top-6 백테스트), `unsupervised.py` (PCA·군집·t-SNE). Extended
      features.py (number_panel), stats_tests.py (cohens_w, chi_square_power).
- [x] **Phase C** — 10-notebook curriculum 00–09 (bilingual, heavy markdown, LaTeX, pitfalls);
      pension moved to 02 as independence-vs-dependence contrast; all execute clean.
- [x] **Phase D** — webapp `/api/report` extended with probability/bayesian/randomness/ml/
      unsupervised; `static_report/` grown to 9-section report + KaTeX + expandable details +
      "score any ticket" widget + theme toggle. Verified via headless Edge (light + dark).
- [x] **Phase E** — README as curriculum index; tests for probability/bayesian/randomness (26 total);
      final verification.

*Status: v2 COMPLETE. Both themes verified, 26 tests pass, 10 notebooks execute clean, app boots.*
