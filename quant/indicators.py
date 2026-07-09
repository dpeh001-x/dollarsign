"""Hand-rolled technical indicators on OHLCV pandas DataFrames.

All functions take a DataFrame with lowercase columns: open, high, low, close, volume.
They return a Series (or DataFrame for multi-output indicators like Bollinger).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def returns(close: pd.Series, log: bool = False) -> pd.Series:
    if log:
        return np.log(close / close.shift(1))
    return close.pct_change()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing (EMA with alpha = 1/length)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def vwap(df: pd.DataFrame, reset_daily: bool = False) -> pd.Series:
    """Volume-weighted average price. Cumulative across the whole frame
    unless reset_daily=True (then per-trading-day for intraday data)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    if reset_daily:
        groups = df.index.normalize()
        return pv.groupby(groups).cumsum() / df["volume"].groupby(groups).cumsum()
    return pv.cumsum() / df["volume"].cumsum()


def vwap_rolling(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """Rolling VWAP over the trailing `length` bars. Unlike cumulative VWAP,
    this doesn't depend on where the data frame happens to start, so the same
    historical bar always gets the same value."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    return pv.rolling(length).sum() / df["volume"].rolling(length).sum().replace(0, np.nan)


def bbands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands. Returns DataFrame with bb_lower, bb_mid, bb_upper, bb_width."""
    mid = close.rolling(length).mean()
    sd = close.rolling(length).std()
    upper = mid + std * sd
    lower = mid - std * sd
    return pd.DataFrame({
        "bb_lower": lower,
        "bb_mid": mid,
        "bb_upper": upper,
        "bb_width": (upper - lower) / mid,
    })


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line, histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": macd_line - signal_line,
    })


def adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """ADX trend strength. Returns +DI, -DI, ADX."""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    tr = atr(df, length=length) * length  # un-normalize back to TR sum proxy
    # Use Wilder smoothed sums directly:
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_w = true_range.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / atr_w
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / atr_w

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_val})


def realized_vol(close: pd.Series, length: int = 20, annualize: int = 252) -> pd.Series:
    """Rolling annualized standard deviation of log returns."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(length).std() * np.sqrt(annualize)


def rolling_drawdown(close: pd.Series, length: int = 20) -> pd.Series:
    """Worst drawdown within trailing window, as a negative pct."""
    rolling_max = close.rolling(length).max()
    return (close - rolling_max) / rolling_max


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume: cumulative volume signed by daily price direction."""
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def vol_ratio(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """Today's volume relative to trailing mean — volume confirmation signal."""
    avg = df["volume"].rolling(length).mean()
    return df["volume"] / avg.replace(0, np.nan)


def hi52_dist(df: pd.DataFrame) -> pd.Series:
    """Distance from 52-week high as negative pct. 0 = at 52w high, -1 = 100% below."""
    high_52w = df["close"].rolling(252).max()
    return (df["close"] - high_52w) / high_52w.replace(0, np.nan)


def attach_all(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a standard battery of indicators to an OHLCV frame. Returns a copy."""
    out = df.copy()
    out["ret_1"] = returns(out["close"])
    out["ret_5"] = out["close"].pct_change(5)
    out["ret_20"] = out["close"].pct_change(20)
    out["rsi_14"] = rsi(out["close"], 14)
    out["atr_14"] = atr(out, 14)
    out["vwap"] = vwap_rolling(out, 20)  # rolling, not anchored to arbitrary frame start
    out = out.join(bbands(out["close"], 20, 2.0))
    out = out.join(macd(out["close"]))
    out = out.join(adx(out, 14))
    out["vol_20"] = realized_vol(out["close"], 20)
    out["dd_20"] = rolling_drawdown(out["close"], 20)
    out["obv"] = obv(out)
    out["vol_ratio_20"] = vol_ratio(out, 20)
    out["hi52_dist"] = hi52_dist(out)
    return out
