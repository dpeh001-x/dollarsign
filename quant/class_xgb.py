"""Asset-class specialized XGBoost models.

For each asset class (equities, sector ETFs, broad ETFs, macro/intl/bond
ETFs, crypto), pool feature/target rows from all symbols in the class
into one training set. Train ONE XGBoost per class. Each per-step training
sees ~5000 samples (e.g., 7 equity symbols × 750-day window) instead of
~500 in the per-symbol model — better generalization on within-class
patterns.

Critical: leakage control. At cutoff date T, training rows are filtered
to those whose forward_date (date + horizon trading days) is also < T —
otherwise the target depends on data at or after T, leaking the future.

Same regime/feature pipeline as ml.py; same WF cadence (train_size=750,
step_size=60). Per-symbol regime model is fit on each symbol's own data;
the pooled model just consumes the regime feature like any other.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ml
import indicators
import regime
from data import sources

logger = logging.getLogger(__name__)


CLASSES: dict[str, list[str]] = {
    "equities":   ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    "broad_etf":  ["SPY", "QQQ", "IWM", "DIA"],
    "sector_etf": ["XLF", "XLE", "XLK", "XLV", "XLY", "XLP", "XLI"],
    "macro_etf":  ["EFA", "EEM", "TLT", "GLD", "USO"],
    "crypto":     ["BTC-USD", "ETH-USD"],
}

SYMBOL_TO_CLASS: dict[str, str] = {
    sym: cls for cls, syms in CLASSES.items() for sym in syms
}


def classify_symbol(symbol: str) -> str | None:
    return SYMBOL_TO_CLASS.get(symbol.upper())


def prepare_symbol(symbol: str, start: str) -> pd.DataFrame:
    """Load → indicators → regime. Per-symbol regime model (consistent with ml.py)."""
    df = sources.get_bars(symbol, start, use_cache=True)
    df = indicators.attach_all(df)
    df = regime.label(df)
    return df


def build_pooled_features(
    symbol_dfs: dict[str, pd.DataFrame],
    horizon: int = 10,
) -> pd.DataFrame:
    """Stack per-symbol features. Each row carries date, symbol, forward_date, features, target.

    forward_date is the calendar date whose close is used in the row's target.
    We use positional shift on each symbol's own index (handles holidays).
    """
    rows: list[pd.DataFrame] = []
    for sym, df in symbol_dfs.items():
        feats = ml.make_features(df, horizon=horizon).copy()
        feats["symbol"] = sym
        feats["date"] = feats.index
        # forward_date: the date the target's close is taken from
        idx_series = pd.Series(df.index, index=df.index)
        feats["forward_date"] = idx_series.shift(-horizon).reindex(feats.index).values
        rows.append(feats)
    pooled = pd.concat(rows, ignore_index=True)
    return pooled.sort_values(["date", "symbol"]).reset_index(drop=True)


def walkforward_pool(
    pooled: pd.DataFrame,
    train_size_dates: int = 750,
    step_size_dates: int = 60,
) -> dict[str, pd.Series]:
    """Walk-forward training on pooled data; returns {symbol: P(up) series}.

    Leak-proof: at cutoff date T, training is restricted to rows whose
    forward_date < T (so the target was realized before T).
    """
    feat_cols = list(ml.FEATURE_COLS)

    # Normalize all datetime columns to UTC nanosecond precision
    pooled = pooled.copy()
    pooled["date"] = pd.to_datetime(pooled["date"], utc=True)
    pooled["forward_date"] = pd.to_datetime(pooled["forward_date"], utc=True)

    all_dates = sorted(pooled["date"].dropna().unique())
    n_dates = len(all_dates)

    # Per-symbol probability series, indexed by that symbol's own dates
    out_proba: dict[str, pd.Series] = {}
    for sym in pooled["symbol"].unique():
        sym_dates = pd.DatetimeIndex(sorted(pooled.loc[pooled["symbol"] == sym, "date"].unique()))
        out_proba[sym] = pd.Series(np.nan, index=sym_dates, dtype=float)

    i = train_size_dates
    while i < n_dates:
        cutoff = all_dates[i]
        end_idx = min(i + step_size_dates, n_dates)
        test_dates = set(all_dates[i:end_idx])

        # Strict leakage guard: forward_date must be before cutoff
        train_mask = (
            (pooled["forward_date"] < cutoff)
            & pooled["target"].notna()
            & pooled["forward_date"].notna()
        )
        train = pooled[train_mask].dropna(subset=feat_cols)

        if len(train) < 200:
            i += step_size_dates
            continue

        try:
            model = ml.fit_xgb(train)
        except Exception as e:
            logger.debug("pooled fit failed at %s: %s", cutoff, e)
            i += step_size_dates
            continue

        test = pooled[pooled["date"].isin(test_dates)].dropna(subset=feat_cols)
        if not test.empty:
            X = test[feat_cols].astype(float)
            proba = model.predict_proba(X)[:, 1]
            for ridx, p in zip(test.index, proba):
                row = test.loc[ridx]
                out_proba[row["symbol"]].loc[row["date"]] = p

        i += step_size_dates

    return out_proba


def predict_class(
    class_name: str,
    start: str,
    horizon: int = 10,
    train_size: int = 750,
    step_size: int = 60,
) -> dict[str, pd.Series]:
    """Top-level: prep all symbols in the class, run pooled WF, return per-symbol probas."""
    symbols = CLASSES.get(class_name)
    if not symbols:
        raise KeyError(f"Unknown class: {class_name}")

    symbol_dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            symbol_dfs[sym] = prepare_symbol(sym, start)
        except Exception as e:
            logger.warning("skip %s during class prep: %s", sym, e)

    if not symbol_dfs:
        return {}

    pooled = build_pooled_features(symbol_dfs, horizon=horizon)
    return walkforward_pool(pooled, train_size_dates=train_size, step_size_dates=step_size)


def proba_to_positions(
    proba: pd.Series,
    long_thresh: float = 0.55,
    short_thresh: float = 0.45,
) -> pd.Series:
    pos = pd.Series(0.0, index=proba.index)
    pos[proba > long_thresh] = 1.0
    pos[proba < short_thresh] = -1.0
    return pos


def predict_today_for_class_ensemble(
    class_name: str,
    start: str,
    horizon: int = 10,
    long_thresh: float = 0.55,
    short_thresh: float = 0.45,
    model_names: tuple[str, ...] = ml.ENSEMBLE_MODELS,
) -> dict[str, dict]:
    """Pooled-ensemble version of predict_today_for_class. Trains EACH model in
    `model_names` on the same pooled training data, then averages their P(up)
    for every symbol's latest bar."""
    import models as _models  # avoid circular import at load time

    symbols = CLASSES.get(class_name)
    if not symbols:
        raise KeyError(f"Unknown class: {class_name}")

    symbol_dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            symbol_dfs[sym] = prepare_symbol(sym, start)
        except Exception as e:
            logger.warning("skip %s during pool prep: %s", sym, e)
    if not symbol_dfs:
        return {}

    pooled = build_pooled_features(symbol_dfs, horizon=horizon)
    feat_cols = list(ml.FEATURE_COLS)
    train = pooled.dropna(subset=["target"]).dropna(subset=feat_cols)
    if len(train) < 200:
        raise RuntimeError(f"Not enough pooled training data: {len(train)} rows")

    X = train[feat_cols].astype(float).to_numpy()
    y = train["target"].astype(int).to_numpy()
    fitted: dict[str, object] = {}
    for name in model_names:
        fit_fn = _models.MODELS.get(name)
        if fit_fn is None:
            continue
        try:
            fitted[name] = fit_fn(X, y)
        except Exception as e:
            logger.warning("ensemble pool: %s fit failed: %s", name, e)

    if not fitted:
        raise RuntimeError("No ensemble models trained successfully")

    # Importance from XGBoost specifically (for the UI chart)
    xgb_model = fitted.get("xgboost")
    importance = (
        pd.Series(xgb_model.feature_importances_, index=feat_cols).sort_values(ascending=False).head(8).to_dict()
        if xgb_model is not None
        else {}
    )

    out: dict[str, dict] = {}
    for sym in symbol_dfs:
        sym_rows = pooled[pooled["symbol"] == sym].dropna(subset=feat_cols)
        if sym_rows.empty:
            continue
        latest = sym_rows.iloc[-1]
        Xrow = latest[feat_cols].astype(float).to_numpy().reshape(1, -1)

        probas = []
        for _, model in fitted.items():
            try:
                probas.append(float(model.predict_proba(Xrow)[0, 1]))
            except Exception:
                pass
        if not probas:
            continue

        avg = float(np.mean(probas))
        signal = "long" if avg > long_thresh else "short" if avg < short_thresh else "flat"
        out[sym] = {
            "proba": avg,
            "signal": signal,
            "confidence": abs(avg - 0.5) * 2,
            "as_of": pd.to_datetime(latest["date"]),
            "class": class_name,
            "n_symbols_pooled": len(symbol_dfs),
            "n_training_rows": len(train),
            "n_models": len(probas),
            "model_names": tuple(fitted.keys()),
            "importance": importance,
        }
    return out


def predict_class_ensemble(
    class_name: str,
    start: str,
    horizon: int = 10,
    train_size: int = 750,
    step_size: int = 60,
    model_names: tuple[str, ...] = ml.ENSEMBLE_MODELS,
) -> dict[str, pd.Series]:
    """Pooled-ensemble walk-forward. At each step, trains every model in
    `model_names` on the pooled training data, then uses the average of
    their probabilities as the prediction for each symbol's test bars."""
    import models as _models

    symbols = CLASSES.get(class_name)
    if not symbols:
        raise KeyError(f"Unknown class: {class_name}")

    symbol_dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            symbol_dfs[sym] = prepare_symbol(sym, start)
        except Exception as e:
            logger.warning("skip %s during pool prep: %s", sym, e)
    if not symbol_dfs:
        return {}

    pooled = build_pooled_features(symbol_dfs, horizon=horizon).copy()
    pooled["date"] = pd.to_datetime(pooled["date"], utc=True)
    pooled["forward_date"] = pd.to_datetime(pooled["forward_date"], utc=True)

    feat_cols = list(ml.FEATURE_COLS)
    all_dates = sorted(pooled["date"].dropna().unique())
    n_dates = len(all_dates)

    out_proba: dict[str, pd.Series] = {}
    for sym in pooled["symbol"].unique():
        sym_dates = pd.DatetimeIndex(sorted(pooled.loc[pooled["symbol"] == sym, "date"].unique()))
        out_proba[sym] = pd.Series(np.nan, index=sym_dates, dtype=float)

    i = train_size
    while i < n_dates:
        cutoff = all_dates[i]
        end_idx = min(i + step_size, n_dates)
        test_dates = set(all_dates[i:end_idx])

        train_mask = (
            (pooled["forward_date"] < cutoff)
            & pooled["target"].notna()
            & pooled["forward_date"].notna()
        )
        train = pooled[train_mask].dropna(subset=feat_cols)
        if len(train) < 200:
            i += step_size
            continue

        X_train = train[feat_cols].astype(float).to_numpy()
        y_train = train["target"].astype(int).to_numpy()
        if len(np.unique(y_train)) < 2:
            i += step_size
            continue

        test = pooled[pooled["date"].isin(test_dates)].dropna(subset=feat_cols)
        if test.empty:
            i += step_size
            continue
        X_test = test[feat_cols].astype(float).to_numpy()

        model_probas: list[np.ndarray] = []
        for name in model_names:
            fit_fn = _models.MODELS.get(name)
            if fit_fn is None:
                continue
            try:
                m = fit_fn(X_train, y_train)
                model_probas.append(m.predict_proba(X_test)[:, 1])
            except Exception as e:
                logger.debug("ens pool WF %s at %s: %s", name, cutoff, e)

        if model_probas:
            avg = np.mean(model_probas, axis=0)
            for ridx, p in zip(test.index, avg):
                row = test.loc[ridx]
                out_proba[row["symbol"]].loc[row["date"]] = float(p)

        i += step_size

    return out_proba


# Classes where universe-wide eval (eval_class_pooled.py) showed pooling
# beats per-symbol training. Used by the Streamlit app to decide whether
# to spend the time training a pooled model for the queried symbol.
POOL_BENEFITS: set[str] = {"equities", "broad_etf", "crypto"}


def predict_today_for_class(
    class_name: str,
    start: str,
    horizon: int = 10,
    long_thresh: float = 0.55,
    short_thresh: float = 0.45,
) -> dict[str, dict]:
    """Train one pooled model on all available class data, predict TODAY's bar
    for every symbol in the class. Returns {symbol: prediction_dict}.

    Single training pass — caller can lookup any symbol in the class from
    the result. Drops rows where target would extend past the last bar
    (target NaN), so latest bars don't leak into training.
    """
    symbols = CLASSES.get(class_name)
    if not symbols:
        raise KeyError(f"Unknown class: {class_name}")

    symbol_dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            symbol_dfs[sym] = prepare_symbol(sym, start)
        except Exception as e:
            logger.warning("skip %s during pool prep: %s", sym, e)

    if not symbol_dfs:
        return {}

    pooled = build_pooled_features(symbol_dfs, horizon=horizon)
    feat_cols = list(ml.FEATURE_COLS)
    train = pooled.dropna(subset=["target"]).dropna(subset=feat_cols)
    if len(train) < 200:
        raise RuntimeError(
            f"Not enough pooled training data for {class_name}: {len(train)} rows"
        )

    model = ml.fit_xgb(train)
    importance = (
        pd.Series(model.feature_importances_, index=feat_cols)
        .sort_values(ascending=False)
        .head(8)
        .to_dict()
    )

    out: dict[str, dict] = {}
    for sym in symbol_dfs:
        sym_rows = pooled[pooled["symbol"] == sym].dropna(subset=feat_cols)
        if sym_rows.empty:
            continue
        latest = sym_rows.iloc[-1]
        feats_arr = latest[feat_cols].astype(float).to_numpy().reshape(1, -1)
        proba = float(model.predict_proba(feats_arr)[0, 1])

        if proba > long_thresh:
            signal = "long"
        elif proba < short_thresh:
            signal = "short"
        else:
            signal = "flat"

        out[sym] = {
            "proba": proba,
            "signal": signal,
            "confidence": abs(proba - 0.5) * 2,
            "as_of": pd.to_datetime(latest["date"]),
            "class": class_name,
            "n_symbols_pooled": len(symbol_dfs),
            "n_training_rows": len(train),
            "importance": importance,
        }
    return out
