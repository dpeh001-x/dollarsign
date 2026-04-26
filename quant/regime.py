"""Market regime classifier.

Unsupervised K-means over rolling features (return, vol, drawdown, ADX trend
strength). Clusters are auto-labeled into one of {trending_up, trending_down,
ranging, volatile_crash} based on their centroid characteristics.

This is *descriptive*, not predictive — it tells you what regime each bar
sits in based on the trailing window. Use it as a state filter in strategies
("only enter breakouts when regime == trending_up"), not as a price forecast.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

REGIME_FEATURES = ["ret_20", "vol_20", "dd_20", "adx", "rsi_14"]


@dataclass
class RegimeModel:
    kmeans: KMeans
    scaler: StandardScaler
    label_map: dict[int, str]
    feature_cols: list[str]

    def predict(self, df: pd.DataFrame) -> pd.Series:
        feats = df[self.feature_cols].copy()
        valid = feats.dropna()
        scaled = self.scaler.transform(valid.to_numpy())
        clusters = self.kmeans.predict(scaled)
        out = pd.Series(index=df.index, dtype="object")
        out.loc[valid.index] = [self.label_map[c] for c in clusters]
        return out


def _label_clusters(km: KMeans, scaler: StandardScaler, feature_cols: list[str]) -> dict[int, str]:
    """Inspect cluster centroids in original units and assign human labels.

    Heuristic: of the 4 clusters, label by ranking on each feature dim:
      - highest vol + most negative dd → "volatile_crash"
      - of remaining: highest +ret_20 + high adx → "trending_up"
      - of remaining: most negative ret_20 + high adx → "trending_down"
      - whatever's left → "ranging"
    """
    centroids = scaler.inverse_transform(km.cluster_centers_)
    cdf = pd.DataFrame(centroids, columns=feature_cols)

    available = set(range(len(cdf)))
    label_map: dict[int, str] = {}

    # 1. Volatile crash: highest vol_20, most negative dd_20
    score = cdf["vol_20"] - cdf["dd_20"]  # both should be high → high vol, deep dd
    crash_idx = score.idxmax()
    label_map[crash_idx] = "volatile_crash"
    available.remove(crash_idx)

    # 2. Trending up: max ret_20 among remaining, weighted by adx
    sub = cdf.loc[list(available)]
    trend_up_idx = (sub["ret_20"] + 0.001 * sub["adx"]).idxmax()
    label_map[trend_up_idx] = "trending_up"
    available.remove(trend_up_idx)

    # 3. Trending down: min ret_20 among remaining
    sub = cdf.loc[list(available)]
    trend_dn_idx = (sub["ret_20"] - 0.001 * sub["adx"]).idxmin()
    label_map[trend_dn_idx] = "trending_down"
    available.remove(trend_dn_idx)

    # 4. Whatever's left: ranging
    for idx in available:
        label_map[idx] = "ranging"

    return label_map


def fit(df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> RegimeModel:
    """Fit a regime classifier on a frame that already has indicators attached."""
    missing = [c for c in REGIME_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required indicator columns: {missing}. Run indicators.attach_all() first.")

    feats = df[REGIME_FEATURES].dropna()
    if len(feats) < n_clusters * 5:
        raise ValueError(f"Not enough data for {n_clusters}-cluster fit: have {len(feats)} rows")

    scaler = StandardScaler()
    X = scaler.fit_transform(feats.to_numpy())

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(X)

    label_map = _label_clusters(km, scaler, REGIME_FEATURES)
    return RegimeModel(kmeans=km, scaler=scaler, label_map=label_map, feature_cols=REGIME_FEATURES)


def label(df: pd.DataFrame, model: RegimeModel | None = None) -> pd.DataFrame:
    """Attach a 'regime' column. Fits a fresh model if one isn't supplied."""
    if model is None:
        model = fit(df)
    out = df.copy()
    out["regime"] = model.predict(out)
    return out


def regime_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-regime summary: count, mean fwd return, mean vol, % of time."""
    if "regime" not in df.columns:
        raise ValueError("DataFrame must have a 'regime' column. Call regime.label() first.")

    fwd_ret = df["close"].pct_change().shift(-1)
    grouped = df.assign(fwd_ret=fwd_ret).groupby("regime")
    stats = pd.DataFrame({
        "count": grouped.size(),
        "pct_time": grouped.size() / len(df.dropna(subset=["regime"])) * 100,
        "mean_fwd_ret_bps": grouped["fwd_ret"].mean() * 10000,
        "mean_vol_ann": grouped["vol_20"].mean(),
        "mean_adx": grouped["adx"].mean(),
        "mean_dd": grouped["dd_20"].mean(),
    })
    return stats.sort_values("pct_time", ascending=False).round(3)
