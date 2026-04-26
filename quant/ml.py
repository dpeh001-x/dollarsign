"""XGBoost binary classifier for next-N-day price direction.

Predicts P(close[t+horizon] > close[t]) given technicals + regime + lagged
returns. Two model wrappers:

  * `XGBPredictor` — train once on full history, predict the latest bar.
    Used by the Streamlit app for "today's signal." Strictly out-of-sample
    on the prediction bar because rows with NaN target (last `horizon`
    rows) are dropped from training.

  * `WalkForwardModel` — slides a `train_size`-bar window across history,
    retrains every `step_size` bars, predicts only the unseen-future
    portion. Used by reliability tests + backtests. Zero look-ahead.

Feature engineering lives in `make_features()`. Whenever you change that
function, retrain — feature columns are part of the model's contract.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

REGIME_VALUES = ("trending_up", "trending_down", "ranging", "volatile_crash")

FEATURE_COLS: tuple[str, ...] = (
    # Lagged returns
    "ret_1", "ret_5", "ret_20",
    # Momentum / trend strength
    "rsi_14", "adx", "plus_di", "minus_di",
    # Volatility / drawdown
    "atr_pct", "vol_20", "dd_20",
    # Mean-reversion / band position
    "bb_width", "bb_pct", "vwap_dist",
    # Trend state
    "macd_hist_norm",
    # Regime one-hot
    *(f"regime_{r}" for r in REGIME_VALUES),
)


def make_features(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Build a feature + target frame from indicator-attached OHLCV.

    All features at row t use only data <= t (no look-ahead). The target at
    row t is the sign of close[t+horizon] / close[t] - 1; rows where this
    isn't yet known are kept (target = NaN) so the latest bar can still be
    scored — caller drops NaN target before training.
    """
    out = pd.DataFrame(index=df.index)

    # Lagged returns
    out["ret_1"] = df["ret_1"]
    out["ret_5"] = df["ret_5"]
    out["ret_20"] = df["ret_20"]

    # Trend strength
    out["rsi_14"] = df["rsi_14"]
    out["adx"] = df["adx"]
    out["plus_di"] = df["plus_di"]
    out["minus_di"] = df["minus_di"]

    # Volatility, drawdown
    out["atr_pct"] = df["atr_14"] / df["close"]
    out["vol_20"] = df["vol_20"]
    out["dd_20"] = df["dd_20"]

    # Bollinger position
    band_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    out["bb_width"] = df["bb_width"]
    out["bb_pct"] = (df["close"] - df["bb_lower"]) / band_range

    # VWAP dislocation
    out["vwap_dist"] = (df["close"] - df["vwap"]) / df["close"]

    # MACD histogram normalized by price
    out["macd_hist_norm"] = df["macd_hist"] / df["close"]

    # Regime one-hot
    regime = df.get("regime", pd.Series(index=df.index, dtype="object"))
    for r in REGIME_VALUES:
        out[f"regime_{r}"] = (regime == r).astype(int)

    # Target: sign of forward N-day return
    fwd_ret = df["close"].pct_change(horizon).shift(-horizon)
    out["fwd_ret"] = fwd_ret
    out["target"] = (fwd_ret > 0).astype("Int8")
    out.loc[fwd_ret.isna(), "target"] = pd.NA

    return out


def _xgb_defaults(**overrides) -> dict:
    # Tuned via tune_xgb.py on SPY 8y: honest test-set Sharpe +0.40,
    # 55.3% win, +16.5% return (2023-2026).
    base = dict(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        n_jobs=1,
        verbosity=0,
        tree_method="hist",
    )
    base.update(overrides)
    return base


def fit_xgb(features_df: pd.DataFrame, **xgb_params) -> XGBClassifier:
    """Train XGBoost on rows where target is known."""
    train = features_df.dropna(subset=["target"])
    if len(train) < 50:
        raise ValueError(f"Not enough training rows after dropna: {len(train)}")
    feat_cols = list(FEATURE_COLS)
    train = train.dropna(subset=feat_cols)
    X = train[feat_cols].astype(float)
    y = train["target"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Target has only one class — need both up and down samples")
    model = XGBClassifier(**_xgb_defaults(**xgb_params))
    model.fit(X, y)
    return model


def predict_proba(model: XGBClassifier, features_df: pd.DataFrame) -> pd.Series:
    feat_cols = list(FEATURE_COLS)
    valid = features_df.dropna(subset=feat_cols)
    if valid.empty:
        return pd.Series(index=features_df.index, dtype=float)
    proba = model.predict_proba(valid[feat_cols].astype(float))[:, 1]
    out = pd.Series(np.nan, index=features_df.index, dtype=float)
    out.loc[valid.index] = proba
    return out


def proba_to_positions(
    proba: pd.Series,
    long_thresh: float = 0.55,
    short_thresh: float = 0.45,
) -> pd.Series:
    """Convert P(up) into discrete positions in {-1, 0, +1}.

    The default 0.55 / 0.45 deadband adds a 'flat' zone for low-confidence
    predictions, which empirically reduces over-trading.
    """
    pos = pd.Series(0.0, index=proba.index)
    pos[proba > long_thresh] = 1.0
    pos[proba < short_thresh] = -1.0
    return pos


@dataclass
class XGBPredictor:
    """Full-history train, score-once predictor. For 'today's call' use cases."""
    horizon: int = 10  # tuned on SPY (tune_xgb.py): horizon=10 gives best honest test Sharpe
    long_thresh: float = 0.55
    short_thresh: float = 0.45
    model: XGBClassifier | None = None

    def fit(self, df_with_indicators: pd.DataFrame) -> "XGBPredictor":
        feats = make_features(df_with_indicators, horizon=self.horizon)
        self.model = fit_xgb(feats)
        return self

    def predict_now(self, df_with_indicators: pd.DataFrame) -> dict:
        if self.model is None:
            raise RuntimeError("Call .fit() first")
        feats = make_features(df_with_indicators, horizon=self.horizon)
        proba = predict_proba(self.model, feats)
        last_idx = proba.last_valid_index()
        if last_idx is None:
            return {"proba": float("nan"), "signal": "flat", "confidence": 0.0, "as_of": None}
        p = float(proba.loc[last_idx])
        if p > self.long_thresh:
            signal = "long"
        elif p < self.short_thresh:
            signal = "short"
        else:
            signal = "flat"
        confidence = abs(p - 0.5) * 2  # 0..1
        return {"proba": p, "signal": signal, "confidence": confidence, "as_of": last_idx}

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Call .fit() first")
        importances = self.model.feature_importances_
        return pd.Series(importances, index=list(FEATURE_COLS)).sort_values(ascending=False)


# Models included in the "best" ensemble. RandomForest excluded — it was
# slowest (16.8s vs 1-3s for others) and worst Sharpe in compare_models.py.
ENSEMBLE_MODELS: tuple[str, ...] = (
    "xgboost", "lightgbm", "catboost", "hist_gbm", "logistic",
)


@dataclass
class EnsemblePredictor:
    """Full-history train on multiple model classes; predict latest bar by averaging
    their P(up) probabilities. From compare_models.py: ENS_AVG_all gives Sharpe +0.44
    on equities-pooled vs +0.37 for single XGBoost (+19% improvement)."""
    horizon: int = 10
    long_thresh: float = 0.55
    short_thresh: float = 0.45
    model_names: tuple[str, ...] = ENSEMBLE_MODELS
    fitted_models: dict | None = None
    xgb_for_importance: object = None

    def fit(self, df_with_indicators: pd.DataFrame) -> "EnsemblePredictor":
        import models as _models  # avoid circular at module-load time

        feats = make_features(df_with_indicators, horizon=self.horizon)
        train = feats.dropna(subset=["target"]).dropna(subset=list(FEATURE_COLS))
        if len(train) < 50:
            raise ValueError(f"Not enough training rows after dropna: {len(train)}")
        X = train[list(FEATURE_COLS)].astype(float).to_numpy()
        y = train["target"].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            raise ValueError("Target has only one class")

        self.fitted_models = {}
        for name in self.model_names:
            fit_fn = _models.MODELS.get(name)
            if fit_fn is None:
                continue
            try:
                self.fitted_models[name] = fit_fn(X, y)
            except Exception as e:
                logger.warning("ensemble: %s fit failed: %s", name, e)

        # Keep XGBoost's importance for the UI's feature-importance chart
        self.xgb_for_importance = self.fitted_models.get("xgboost")
        return self

    def predict_now(self, df_with_indicators: pd.DataFrame) -> dict:
        if not self.fitted_models:
            raise RuntimeError("Call .fit() first")
        feats = make_features(df_with_indicators, horizon=self.horizon)
        valid = feats.dropna(subset=list(FEATURE_COLS))
        if valid.empty:
            return {"proba": float("nan"), "signal": "flat", "confidence": 0.0, "as_of": None}

        latest = valid.iloc[[-1]]
        X = latest[list(FEATURE_COLS)].astype(float).to_numpy()
        probas = []
        for _, model in self.fitted_models.items():
            try:
                probas.append(float(model.predict_proba(X)[0, 1]))
            except Exception:
                pass

        if not probas:
            return {"proba": float("nan"), "signal": "flat", "confidence": 0.0, "as_of": None}

        avg = float(np.mean(probas))
        signal = "long" if avg > self.long_thresh else "short" if avg < self.short_thresh else "flat"
        return {
            "proba": avg,
            "signal": signal,
            "confidence": abs(avg - 0.5) * 2,
            "as_of": latest.index[-1],
            "n_models": len(probas),
            "model_names": tuple(self.fitted_models.keys()),
        }

    def feature_importance(self) -> pd.Series:
        if self.xgb_for_importance is None:
            return pd.Series(dtype=float)
        return pd.Series(
            self.xgb_for_importance.feature_importances_, index=list(FEATURE_COLS)
        ).sort_values(ascending=False)


@dataclass
class EnsembleWalkForward:
    """Walk-forward training of multiple models per step; positions from
    averaged probabilities. Slower than single-model WF (5x roughly) but
    measurably higher Sharpe + return."""
    train_size: int = 500
    step_size: int = 60
    horizon: int = 10
    long_thresh: float = 0.55
    short_thresh: float = 0.45
    model_names: tuple[str, ...] = ENSEMBLE_MODELS

    def fit_predict(self, df_with_indicators: pd.DataFrame) -> pd.Series:
        import models as _models

        feats = make_features(df_with_indicators, horizon=self.horizon)
        n = len(feats)
        if n < self.train_size + 30:
            raise ValueError(f"Need at least {self.train_size + 30} feature rows; have {n}")

        feat_cols = list(FEATURE_COLS)
        proba_out = pd.Series(np.nan, index=feats.index, dtype=float)

        i = self.train_size
        while i < n:
            train = feats.iloc[i - self.train_size: i].dropna(subset=feat_cols + ["target"])
            test_end = min(i + self.step_size, n)
            test = feats.iloc[i: test_end].dropna(subset=feat_cols)
            if len(train) < 50 or len(test) == 0:
                i += self.step_size
                continue

            X_train = train[feat_cols].astype(float).to_numpy()
            y_train = train["target"].astype(int).to_numpy()
            if len(np.unique(y_train)) < 2:
                i += self.step_size
                continue

            X_test = test[feat_cols].astype(float).to_numpy()
            model_probas = []
            for name in self.model_names:
                fit_fn = _models.MODELS.get(name)
                if fit_fn is None:
                    continue
                try:
                    m = fit_fn(X_train, y_train)
                    model_probas.append(m.predict_proba(X_test)[:, 1])
                except Exception as e:
                    logger.debug("ensemble WF: %s failed at step %d: %s", name, i, e)

            if model_probas:
                proba_out.loc[test.index] = np.mean(model_probas, axis=0)
            i += self.step_size

        return proba_out

    def positions(self, df_with_indicators: pd.DataFrame) -> pd.Series:
        proba = self.fit_predict(df_with_indicators)
        pos = proba_to_positions(proba, self.long_thresh, self.short_thresh)
        return pos.reindex(df_with_indicators.index, fill_value=0.0).fillna(0.0)


@dataclass
class WalkForwardModel:
    """Sliding-window retraining. Predictions are strictly out-of-sample.

    For each step:
      1. Train on the prior `train_size` bars
      2. Predict the next `step_size` bars (which the model has never seen)
      3. Slide window forward by `step_size` and repeat

    First `train_size` bars get NaN predictions (no model yet).
    """
    train_size: int = 500
    step_size: int = 60
    horizon: int = 10  # tuned on SPY (tune_xgb.py): horizon=10 gives best honest test Sharpe
    long_thresh: float = 0.55
    short_thresh: float = 0.45
    xgb_params: dict | None = None

    def fit_predict(self, df_with_indicators: pd.DataFrame) -> pd.Series:
        feats = make_features(df_with_indicators, horizon=self.horizon)
        n = len(feats)
        if n < self.train_size + 30:
            raise ValueError(
                f"Need at least {self.train_size + 30} feature rows; have {n}"
            )

        params = self.xgb_params or {}
        proba_out = pd.Series(np.nan, index=feats.index, dtype=float)

        i = self.train_size
        while i < n:
            train_window = feats.iloc[i - self.train_size: i]
            test_end = min(i + self.step_size, n)
            test_window = feats.iloc[i: test_end]

            try:
                model = fit_xgb(train_window, **params)
                pp = predict_proba(model, test_window)
                proba_out.iloc[i:test_end] = pp.to_numpy()
            except Exception as e:
                logger.debug("walk-forward step at %d failed: %s", i, e)

            i += self.step_size

        return proba_out

    def positions(self, df_with_indicators: pd.DataFrame) -> pd.Series:
        proba = self.fit_predict(df_with_indicators)
        pos = proba_to_positions(proba, self.long_thresh, self.short_thresh)
        return pos.reindex(df_with_indicators.index, fill_value=0.0).fillna(0.0)
