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
