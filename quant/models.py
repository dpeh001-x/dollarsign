"""Adapter functions for alternative classifiers.

Each `fit_*` function takes (X, y) and returns a fitted model that exposes
`.predict_proba(X)` returning P(class=0), P(class=1) per row. This lets the
walk-forward driver swap models without touching its plumbing.

Hyperparameters are tuned to be roughly comparable: ~100 weak learners,
shallow depth (3), modest learning rate (0.05). Linear baseline is
included to confirm gradient boosting is actually adding value.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fit_xgboost(X, y, **overrides):
    from xgboost import XGBClassifier
    params = dict(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=1.0, random_state=42, eval_metric="logloss",
        tree_method="hist", verbosity=0, n_jobs=1,
    )
    params.update(overrides)
    model = XGBClassifier(**params)
    model.fit(X, y)
    return model


def fit_lightgbm(X, y, **overrides):
    from lightgbm import LGBMClassifier
    params = dict(
        n_estimators=100, max_depth=3, num_leaves=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_lambda=1.0, random_state=42, verbosity=-1, n_jobs=1,
    )
    params.update(overrides)
    model = LGBMClassifier(**params)
    model.fit(X, y)
    return model


def fit_catboost(X, y, **overrides):
    from catboost import CatBoostClassifier
    params = dict(
        iterations=100, depth=3, learning_rate=0.05,
        l2_leaf_reg=3.0, random_seed=42, verbose=0,
        thread_count=1, allow_writing_files=False,
    )
    params.update(overrides)
    model = CatBoostClassifier(**params)
    model.fit(X, y)
    return model


def fit_hist_gbm(X, y, **overrides):
    from sklearn.ensemble import HistGradientBoostingClassifier
    params = dict(
        max_iter=100, max_depth=3, learning_rate=0.05,
        l2_regularization=1.0, random_state=42,
    )
    params.update(overrides)
    model = HistGradientBoostingClassifier(**params)
    model.fit(X, y)
    return model


def fit_random_forest(X, y, **overrides):
    from sklearn.ensemble import RandomForestClassifier
    params = dict(
        n_estimators=200, max_depth=8, min_samples_leaf=20,
        random_state=42, n_jobs=1,
    )
    params.update(overrides)
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    return model


def fit_logistic(X, y, **overrides):
    """Standardize features then fit L2 logistic regression. Returns a Pipeline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    params = dict(C=1.0, max_iter=2000, random_state=42)
    params.update(overrides)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(**params)),
    ])
    pipe.fit(X, y)
    return pipe


# Registry — order matters for table output
MODELS: dict[str, callable] = {
    "xgboost":       fit_xgboost,
    "lightgbm":      fit_lightgbm,
    "catboost":      fit_catboost,
    "hist_gbm":      fit_hist_gbm,
    "random_forest": fit_random_forest,
    "logistic":      fit_logistic,
}


def walk_forward_predict(
    features_df: pd.DataFrame,
    fit_fn,
    feature_cols: list[str],
    train_size: int = 500,
    step_size: int = 60,
    **fit_kwargs,
) -> pd.Series:
    """Generic walk-forward driver. Returns P(up) series aligned to features_df.index.

    First `train_size` rows get NaN (no model trained yet). Strict OOS:
    each prediction comes from a model trained on rows strictly before it.
    """
    n = len(features_df)
    if n < train_size + 30:
        raise ValueError(f"Need >= {train_size + 30} feature rows; have {n}")

    proba_out = pd.Series(np.nan, index=features_df.index, dtype=float)

    i = train_size
    while i < n:
        train = features_df.iloc[i - train_size: i].dropna(subset=feature_cols + ["target"])
        test_end = min(i + step_size, n)
        test = features_df.iloc[i: test_end].dropna(subset=feature_cols)

        if len(train) < 50 or len(test) == 0:
            i += step_size
            continue

        try:
            X_train = train[feature_cols].astype(float).to_numpy()
            y_train = train["target"].astype(int).to_numpy()
            if len(np.unique(y_train)) < 2:
                i += step_size
                continue

            model = fit_fn(X_train, y_train, **fit_kwargs)
            X_test = test[feature_cols].astype(float).to_numpy()
            proba = model.predict_proba(X_test)[:, 1]
            proba_out.loc[test.index] = proba
        except Exception as e:
            logger.debug("WF step at %d failed: %s", i, e)

        i += step_size

    return proba_out
