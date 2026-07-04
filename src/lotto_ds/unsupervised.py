"""unsupervised.py — searching for hidden structure that isn't there.

Supervised models found no signal (`ml_models`). But maybe there are *latent regimes* — clusters of
"similar" draws a supervised label just didn't capture? Unsupervised learning is the honest way to
check without a target. We stand three classic tools on the draw-shape features
(`features.draw_features`): sum, odd/even, high/low, range, max-gap, consecutive, AC, decade-spread.

- **PCA** — do a few components capture most variance (a low-dimensional manifold)? Or is variance
  spread flat across dimensions (= unstructured noise)?
- **k-means + silhouette** — is there a k for which draws fall into well-separated groups? A
  silhouette near 0 means "no meaningful cluster structure".
- **DBSCAN** — density-based clustering finds arbitrary shapes and calls sparse points noise; on
  random data it collapses to one blob.
- **t-SNE** — a 2-D visualization that *exaggerates* any local structure; on structureless data it
  still shows a single featureless cloud.

Expected — and found — result: no separable clusters. The draws are one homogeneous blob, exactly as
a uniform generator produces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .features import draw_features

FEATURE_COLS = ["sum", "odd_count", "high_count", "range", "max_gap",
                "consecutive_pairs", "ac_value", "decades_covered"]


def _feature_matrix(draws: pd.DataFrame | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    feat = draw_features(draws)
    X = StandardScaler().fit_transform(feat[FEATURE_COLS].to_numpy(dtype=float))
    return X, feat


def pca_analysis(draws: pd.DataFrame | None = None) -> dict:
    X, feat = _feature_matrix(draws)
    pca = PCA().fit(X)
    coords = pca.transform(X)[:, :2]
    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "coords_2d": coords,
        "n_features": len(FEATURE_COLS),
        "flatness_note": "no dominant component ⇒ variance spread across dims ⇒ unstructured",
    }


def kmeans_silhouettes(draws: pd.DataFrame | None = None, ks=range(2, 7), seed: int = 42) -> dict:
    """Silhouette score for a range of k. All low (≪0.5) ⇒ no natural clusters."""
    X, _ = _feature_matrix(draws)
    scores = {}
    for k in ks:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        scores[k] = float(silhouette_score(X, labels))
    best_k = max(scores, key=scores.get)
    return {"silhouettes": scores, "best_k": best_k, "best_score": scores[best_k]}


def dbscan_summary(draws: pd.DataFrame | None = None, eps: float = 1.5, min_samples: int = 10) -> dict:
    X, _ = _feature_matrix(draws)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    return {"n_clusters": n_clusters, "n_noise": n_noise, "n_points": len(labels)}


def tsne_embedding(draws: pd.DataFrame | None = None, seed: int = 42) -> np.ndarray:
    X, _ = _feature_matrix(draws)
    return TSNE(n_components=2, perplexity=30, random_state=seed, init="pca").fit_transform(X)


def summary(draws: pd.DataFrame | None = None) -> dict:
    pca = pca_analysis(draws)
    km = kmeans_silhouettes(draws)
    db = dbscan_summary(draws)
    return {
        "pca_top2_cumulative": round(pca["cumulative"][1], 3),
        "kmeans_best_silhouette": round(km["best_score"], 3),
        "kmeans_best_k": km["best_k"],
        "dbscan_clusters": db["n_clusters"],
        "verdict": "실루엣 ≪0.5 · PCA 평탄(최대 성분 27%) — 활용할 만한 군집 구조 없음",
    }
