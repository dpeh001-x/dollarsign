"""Example strategies that produce a position series in [-1, 1]."""
from __future__ import annotations

import numpy as np
import pandas as pd


def regime_aware_breakout(
    df: pd.DataFrame,
    lookback: int = 20,
    rsi_overbought: float = 70,
    rsi_oversold: float = 30,
) -> pd.Series:
    """Donchian-style breakout, gated by regime.

    Long when:  close > rolling_max(lookback) AND regime == trending_up AND RSI < overbought
    Short when: close < rolling_min(lookback) AND regime == trending_down AND RSI > oversold
    Flat in ranging or volatile_crash regimes (avoid chop and tail risk).
    """
    if "regime" not in df.columns:
        raise ValueError("Frame needs 'regime' column. Run regime.label() first.")

    high_n = df["high"].rolling(lookback).max().shift(1)
    low_n = df["low"].rolling(lookback).min().shift(1)

    long_signal = (
        (df["close"] > high_n)
        & (df["regime"] == "trending_up")
        & (df["rsi_14"] < rsi_overbought)
    )
    short_signal = (
        (df["close"] < low_n)
        & (df["regime"] == "trending_down")
        & (df["rsi_14"] > rsi_oversold)
    )

    pos = pd.Series(0.0, index=df.index)
    pos[long_signal] = 1.0
    pos[short_signal] = -1.0

    # Hold position until opposite signal or regime flip back to ranging/crash
    pos = pos.replace(0, np.nan).ffill().fillna(0)
    pos[df["regime"].isin(["volatile_crash", "ranging"])] = 0
    return pos


def rsi_mean_reversion(df: pd.DataFrame, oversold: float = 30, overbought: float = 70) -> pd.Series:
    """Classic RSI mean reversion. Long when oversold, exit at midline. Mirror for short.

    Only operates in 'ranging' regime if it exists in the frame; otherwise unconditional.
    """
    rsi = df["rsi_14"]
    long_entry = rsi < oversold
    long_exit = rsi >= 50
    short_entry = rsi > overbought
    short_exit = rsi <= 50

    pos = pd.Series(0.0, index=df.index)
    state = 0
    for i, ts in enumerate(df.index):
        if state == 0:
            if long_entry.iloc[i]:
                state = 1
            elif short_entry.iloc[i]:
                state = -1
        elif state == 1 and long_exit.iloc[i]:
            state = 0
        elif state == -1 and short_exit.iloc[i]:
            state = 0
        pos.iloc[i] = state

    if "regime" in df.columns:
        pos[df["regime"] != "ranging"] = 0

    return pos


def ma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """Simple moving average crossover."""
    ma_fast = df["close"].rolling(fast).mean()
    ma_slow = df["close"].rolling(slow).mean()
    pos = pd.Series(0.0, index=df.index)
    pos[ma_fast > ma_slow] = 1.0
    pos[ma_fast < ma_slow] = -1.0
    return pos


def xgb_walk_forward(
    df: pd.DataFrame,
    horizon: int = 5,
    train_size: int = 500,
    step_size: int = 60,
    long_thresh: float = 0.55,
    short_thresh: float = 0.45,
) -> pd.Series:
    """ML strategy: walk-forward XGBoost classifier predicts next-N-day direction.

    Slides a 500-bar training window across history, retrains every 60 bars,
    predicts only out-of-sample. Positions in {-1, 0, +1} based on whether
    P(up) crosses the long/short thresholds.
    """
    import ml
    model = ml.WalkForwardModel(
        train_size=train_size,
        step_size=step_size,
        horizon=horizon,
        long_thresh=long_thresh,
        short_thresh=short_thresh,
    )
    return model.positions(df)
