"""Macro overlay data: VIX fear gauge, fetched from yfinance and parquet-cached.

`get_vix(start, end)` returns a daily DataFrame aligned to trading dates with:
  - vix_level : VIX close (implied vol of S&P 500 options)
  - vix_change: 5-day pct change in VIX (regime shift signal)

The cache key includes the date range so stale entries are avoided.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
_CACHE_DIR.mkdir(exist_ok=True)


def get_vix(start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """Return daily VIX DataFrame with vix_level and vix_change columns.

    Index is DatetimIndex (UTC). Dates without VIX (weekends / holidays) are
    dropped — callers should left-join on their own price index and ffill.
    """
    cache_file = _CACHE_DIR / f"VIX_{start}_{end}.parquet"
    if use_cache and cache_file.exists():
        logger.info("Cache hit: %s", cache_file.name)
        return pd.read_parquet(cache_file)

    import yfinance as yf
    raw = yf.download("^VIX", start=start, end=end, interval="1d", progress=False,
                      auto_adjust=True, threads=False)
    if raw.empty:
        logger.warning("VIX download returned empty for %s–%s", start, end)
        return pd.DataFrame(columns=["vix_level", "vix_change"])

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    raw.index.name = "date"

    out = pd.DataFrame(index=raw.index)
    out["vix_level"] = raw["close"]
    out["vix_change"] = raw["close"].pct_change(5)

    out = out.dropna(subset=["vix_level"])

    if use_cache:
        out.to_parquet(cache_file)
    return out


def get_market_context(start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """VIX + SPY benchmark context in one daily frame.

    Columns: vix_level, vix_change, spy_ret_5, spy_ret_20.
    The SPY columns give every model a read on broad-market momentum, and
    let make_features() compute per-symbol relative strength vs the market.
    """
    cache_file = _CACHE_DIR / f"MKTCTX_{start}_{end}.parquet"
    if use_cache and cache_file.exists():
        logger.info("Cache hit: %s", cache_file.name)
        return pd.read_parquet(cache_file)

    vix = get_vix(start, end, use_cache=use_cache)

    import yfinance as yf
    spy = yf.download("SPY", start=start, end=end, interval="1d", progress=False,
                      auto_adjust=True, threads=False)
    if spy.empty:
        logger.warning("SPY download empty; returning VIX-only context")
        return vix

    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.columns = [c.lower() for c in spy.columns]
    if spy.index.tz is None:
        spy.index = spy.index.tz_localize("UTC")

    ctx = pd.DataFrame(index=spy.index)
    ctx["spy_ret_5"] = spy["close"].pct_change(5)
    ctx["spy_ret_20"] = spy["close"].pct_change(20)

    out = ctx.join(vix, how="left").ffill()

    # VIX term structure: VIX / VIX3M. Ratio > 1 (backwardation) = near-term
    # stress; < 1 (contango) = calm. The slope has documented predictive value
    # for 5-20 day equity returns — more than the VIX level alone.
    try:
        v3 = yf.download("^VIX3M", start=start, end=end, interval="1d", progress=False,
                         auto_adjust=True, threads=False)
        if not v3.empty:
            if isinstance(v3.columns, pd.MultiIndex):
                v3.columns = v3.columns.get_level_values(0)
            v3.columns = [c.lower() for c in v3.columns]
            if v3.index.tz is None:
                v3.index = v3.index.tz_localize("UTC")
            v3_close = v3["close"].reindex(out.index).ffill().replace(0, pd.NA)
            out["vix_ts"] = (out["vix_level"] / v3_close).astype(float)
    except Exception as e:
        logger.warning("VIX3M fetch failed; vix_ts omitted: %s", e)

    if use_cache:
        out.to_parquet(cache_file)
    return out
