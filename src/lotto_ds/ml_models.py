"""ml_models.py — 'machine learning done right' on a problem that has no signal.

The legacy app trained an MLP and a Random Forest and presented their picks as an "AI prediction".
That is not wrong because ML is wrong — it is wrong because it was never *evaluated* honestly. This
module does the ML properly and lets the metrics deliver the verdict:

- **Task**: binary classification — *will number k appear in the next draw?* (features from
  `features.number_panel`, strictly leakage-free).
- **Split**: temporal / walk-forward — train on early draws, test on later ones (never shuffle time).
- **Models**: `LogisticRegression` (linear baseline) and `HistGradientBoostingClassifier` (a strong
  modern gradient-boosting learner — the same family as XGBoost/LightGBM, bundled with scikit-learn).
- **Metrics that matter here**: ROC-AUC (chance = 0.50), a calibration curve (are predicted
  probabilities honest?), and permutation feature importance (which features actually help — none do).

The teaching punchline: a well-tuned gradient booster lands at **AUC ≈ 0.50**. When your best model
cannot separate positives from negatives better than a coin, that is not a modeling failure — it is
the correct discovery that the target is unpredictable. SHAP could deepen the importance story; we
use permutation importance to stay dependency-light.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import LOTTO_PICKS, LOTTO_NUMBER_MAX
from .features import PANEL_FEATURES, number_panel

BASE_RATE = LOTTO_PICKS / LOTTO_NUMBER_MAX  # 6/45 — the positive-class prevalence


@dataclass
class MLResult:
    model_name: str
    auc: float
    n_train: int
    n_test: int
    calibration: dict = field(default_factory=dict)   # {"pred": [...], "true": [...]}
    roc: dict = field(default_factory=dict)            # {"fpr": [...], "tpr": [...]}
    importances: dict = field(default_factory=dict)    # {feature: mean_importance}

    def as_row(self) -> dict:
        return {
            "model": self.model_name,
            "AUC": round(self.auc, 4),
            "vs chance (0.50)": round(self.auc - 0.5, 4),
            "n_test": self.n_test,
        }


def _temporal_split(panel: pd.DataFrame, split_frac: float = 0.7):
    draw_nos = np.sort(panel["draw_no"].unique())
    cut = draw_nos[int(len(draw_nos) * split_frac)]
    train = panel[panel["draw_no"] < cut]
    test = panel[panel["draw_no"] >= cut]
    return train, test


def evaluate_models(panel: pd.DataFrame | None = None, split_frac: float = 0.7,
                    seed: int = 42) -> list[MLResult]:
    """Fit the baseline + gradient booster on a temporal split and return honest metrics."""
    if panel is None:
        panel = number_panel()
    train, test = _temporal_split(panel, split_frac)
    Xtr, ytr = train[PANEL_FEATURES].to_numpy(), train["appeared_next"].to_numpy()
    Xte, yte = test[PANEL_FEATURES].to_numpy(), test["appeared_next"].to_numpy()

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=300, random_state=seed),
    }

    results = []
    for name, model in models.items():
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        auc = float(roc_auc_score(yte, proba))
        frac_pos, mean_pred = calibration_curve(yte, proba, n_bins=10, strategy="quantile")
        fpr, tpr, _ = roc_curve(yte, proba)
        # permutation importance (on a capped test subsample for speed)
        idx = np.random.default_rng(seed).choice(len(Xte), size=min(6000, len(Xte)), replace=False)
        perm = permutation_importance(model, Xte[idx], yte[idx], n_repeats=5,
                                      random_state=seed, scoring="roc_auc")
        imps = {f: float(v) for f, v in zip(PANEL_FEATURES, perm.importances_mean)}
        # thin the ROC curve for transport
        step = max(1, len(fpr) // 100)
        results.append(MLResult(
            model_name=name, auc=auc, n_train=len(ytr), n_test=len(yte),
            calibration={"pred": mean_pred.tolist(), "true": frac_pos.tolist()},
            roc={"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist()},
            importances=imps,
        ))
    return results


def topk_backtest(panel: pd.DataFrame | None = None, split_frac: float = 0.7, k: int = LOTTO_PICKS,
                  seed: int = 42) -> dict:
    """Turn the gradient booster into a ticket picker and backtest it honestly.

    On each *test* draw, rank the 45 numbers by the model's predicted appearance probability and
    'buy' the top-k. Count how many actually appeared. If the model had real skill, mean hits would
    exceed the random baseline 6·6/45 = 0.8. It does not.
    """
    if panel is None:
        panel = number_panel()
    train, test = _temporal_split(panel, split_frac)
    Xtr, ytr = train[PANEL_FEATURES].to_numpy(), train["appeared_next"].to_numpy()
    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                           random_state=seed).fit(Xtr, ytr)
    test = test.copy()
    test["proba"] = model.predict_proba(test[PANEL_FEATURES].to_numpy())[:, 1]

    hits = []
    for _, grp in test.groupby("draw_no"):
        topk = grp.nlargest(k, "proba")
        hits.append(int(topk["appeared_next"].sum()))
    hits = np.array(hits, dtype=float)
    rng = np.random.default_rng(seed)
    boot = rng.choice(hits, size=(10000, len(hits)), replace=True).mean(axis=1)
    return {
        "strategy": "ML top-6 (gradient boosting)",
        "mean": float(hits.mean()),
        "ci_lo": float(np.percentile(boot, 2.5)),
        "ci_hi": float(np.percentile(boot, 97.5)),
        "baseline": LOTTO_PICKS * BASE_RATE,   # expected ticket hits under random = 6·6/45 = 0.8
        "n_draws": len(hits),
    }


def summary(panel: pd.DataFrame | None = None) -> dict:
    results = evaluate_models(panel)
    return {
        "base_rate": BASE_RATE,
        "chance_auc": 0.5,
        "models": [r.as_row() for r in results],
        "best_auc": max(r.auc for r in results),
        "verdict": "AUC≈0.5 — 최상의 모델도 동전 던지기와 구별되지 않음",
    }
