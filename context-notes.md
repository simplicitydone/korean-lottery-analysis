# Context Notes — 3004_lot Do-over

Append-only log of decisions and their reasoning, so the next session can pick up without re-deriving.

## 2026-07-05 — Direction set

**Ask:** Transform the lottery *prediction* app into a data-science *study* showcasing
statistical analysis methods and data cleaning.

**User decisions (via clarifying questions):**
- Deliverable: **notebooks + reframed web app** (both).
- Purpose: **portfolio / showcase piece**.
- Language: **bilingual** — Korean narrative, English code/technical terms.
- Prediction: **keep as a debunked hypothesis** — rigorously test whether any method beats
  chance and show it doesn't. The null result is the headline.

**Why this framing:** Lottery draws are engineered to be uniform-random. An honest DS project
does not pretend to predict them; it (a) cleans genuinely messy source data, (b) does proper
EDA, (c) runs inferential tests for uniformity/independence, and (d) demonstrates via
leakage-free backtest that no strategy beats random. That is a stronger portfolio signal than
a fake "AI predictor."

## Data-quality findings (raw lottery.db, profiled 2026-07-05)

- `draw_results`: **1230 lotto draws** (draw_no 1..1230, dates 2002-12-07 .. 2026-06-27).
  Core numeric cols (win1..win6, bonus) are **clean**: no dupes, no seq gaps, no out-of-range,
  no nulls. Dates already ISO `YYYY-MM-DD`.
- **The mess is schema hygiene**, not values:
  - **399 phantom `extra_0..extra_398` columns** from a raw Excel→SQLite dump; 61 hold stray
    scattered values, the rest fully null.
  - Prize columns are **mixed-type**: `prize_1st_total` is TEXT (currency-formatted),
    `prize_2nd_count`/`prize_3rd_*` are REAL, others INTEGER. Need parse → int/float.
  - **8 orphan sheet-dump tables** with generic `col_0..col_N` names and no real schema:
    `number_stats` (150 cols), `consecutive_stats`, `sum_odd_even`, `empty_sheet`,
    `verification`, `number_freq_dist`, `sum_freq_dist`, `prize_calc_logic`. These are Excel
    tabs dumped verbatim — treat as cruft / derive fresh instead.
- `pension_results`: **343 rows across 322 draws**; `is_bonus` flag mixes main + bonus rows.
  Dates ISO from 2020-05-07. (README/updated.md earlier flagged a #310 date fix — already applied.)
- Prediction archive already exists: `prediction_accuracy_v3` (500 rows), `_pension_v3` (18),
  `prediction_accuracy` (10, legacy). Reuse for the backtest reconciliation instead of recomputing.

**Cleaning strategy:** don't mutate raw lottery.db (app.py still depends on it). Read raw,
write a fresh normalized `data/clean/lotto_clean.db` with tidy tables:
`draws`, `draw_numbers` (long form), `prizes`, `pension_draws`. Pipeline lives in
`src/lotto_ds/cleaning.py` so notebooks and (later) the web app share one source of truth.

## Environment

- Global Python 3.12.10 has pandas/numpy/scipy/statsmodels/matplotlib; **missing** sklearn,
  jupyter, nbconvert, seaborn. Created `.venv` with pinned DS deps (`requirements-ds.txt`).
- Windows laptop. `.venv/Scripts/python.exe` is the interpreter. bash=WSL here, so use
  PowerShell/venv python for execution.
- Notebooks authored as jupytext `.py:percent` where convenient, materialized to `.ipynb`,
  executed headless via `nbconvert --execute` for verification.

## Key statistical findings (for README / web app copy)

- **Number uniformity χ²**: χ²=27.9, df=44, **p=0.97** — indistinguishable from uniform (χ² *below*
  df; observed freq spread 136–184 around expected 164 is pure sampling noise).
- **Runs test (odd/even majority)**: z=-1.14, **p=0.25** — no temporal streaking.
- **Ljung–Box on draw sums**: p=0.032 at lag 10 (the ONE marginal hit). Honestly debunked:
  max |acf|=0.074 (r²<0.6%), only significant at lags 10–11 (gone by 12), and no lag clears the
  Bonferroni threshold 0.05/15. → a multiple-comparisons mirage, not signal. Great p-hacking lesson.
- **Hot-number edge (walk-forward, window=50)**: observed mean overlap 0.797 vs random 0.80,
  t=-0.80, **p=0.42** — hot numbers give zero edge.
- **Pension digit uniformity**: all 6 positions p>0.05.
- **Verdict**: both lotteries are statistically indistinguishable from a fair uniform RNG.

## Phases done
- Phase 0 setup, Phase 1 cleaning (nb01, 12 tests), Phase 2 EDA (features.py, nb02),
  Phase 3 hypothesis testing (stats_tests.py, nb03). All notebooks execute clean via nbconvert.

## Status: ALL PHASES COMPLETE (2026-07-05)

Delivered the full do-over in one session:
- Phase 4: `backtest.py` + nb04 — 5 strategies + legacy 19,560-ticket reconciliation, all ≈0.80.
- Phase 5: NEW `webapp.py` (DS report API `/api/report`) + `static_report/index.html` (bilingual
  theme-aware SPA, Chart.js, validated palette). Legacy `app.py`/`static/` untouched. Rendered &
  visually verified via headless Edge screenshots.
- Phase 6: README rewritten as bilingual portfolio landing; requirements-ds.txt pinned;
  data/clean/ gitignored.

Final verification: py_compile OK · 12 tests pass · clean DB rebuilds from scratch · all 4
notebooks execute clean via nbconvert · webapp boots, all endpoints 200.

## v2 Refinement COMPLETE (2026-07-05)

Turned the study into an educational "show-off" curriculum. Legacy prediction app fully deleted
(no password anywhere). Added 5 new analysis modules + 6 new notebooks (now 00–09) + 5 new web
sections. Pension reframed as the independence-vs-dependence probability contrast (notebook 02).

- New modules: probability, randomness, bayesian, ml_models, unsupervised (+ features.number_panel,
  stats_tests.cohens_w/chi_square_power). **Zero new runtime deps** — sklearn/scipy/statsmodels cover all.
- Key results: χ² p=0.97 (w≈0.06, power≈1.0) · randomness 5/6 pass (spectral peak = lag-10 acf dual) ·
  Bayesian 44/45 CIs contain 6/45 · ML AUC 0.49 (importances≈0) · unsupervised silhouette 0.19 ·
  backtest all ≈0.80 incl. GBM top-6 (0.791).
- Web app: 9 sections + KaTeX + expandable details + "score any ticket" widget + theme toggle.
  Verified light AND dark via headless Edge. **Browser blocks ports 5060/5061 (ERR_UNSAFE_PORT)** —
  use 8xxx for screenshots.
- Verification: py_compile OK · 26 tests pass · clean DB rebuilds · all 10 notebooks execute clean ·
  serve.py (waitress) boots, all endpoints 200.

### Gotchas for next session
- Windows console is cp949 → set `PYTHONIOENCODING=utf-8` for any script printing Korean/em-dash.
- No jupyter/sklearn/flask in global python; everything lives in `.venv` (`.venv/Scripts/python.exe`).
- SQLite has no nullable-int → Int64 columns read back as float64; don't assert exact Int64 dtype.
- Visual QA of the web page: headless Edge at
  `/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe --headless=new --virtual-time-budget=9000 --screenshot=...`
- Not yet done (optional future): light-mode visual QA (only dark verified), git commits (worktree
  dirty — left uncommitted for user review), pension EDA depth, deploy.
