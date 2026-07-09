"""Model 2 — Cross-Sectional Momentum + RSI Mean-Reversion overlay.

Ranks a basket of symbols by blended medium-term momentum and flags
short-term overextension with RSI. Two sleeves come out of one table:

  Momentum sleeve (trend-following):
    score = 0.5 * 63-day return + 0.5 * 126-day return, both computed
    with a 5-day skip (price 5 days ago vs price 68/131 days ago) so the
    well-documented short-term reversal effect doesn't contaminate the
    momentum signal. Long the top N, (optionally) short the bottom N.

  Mean-reversion sleeve (buy the dip / fade the rip):
    RSI(14) < 30 while the stock is still above its own 200-day MA
      → "oversold in an uptrend": a dip worth buying.
    RSI(14) > 70 → overbought: don't chase; short candidate in risk-off.

Everything is a hard number — no discretion, no story-chasing.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import indicators  # noqa: E402

logger = logging.getLogger(__name__)

SKIP = 5  # days skipped in momentum lookbacks (short-term reversal guard)


def momentum_row(close: pd.Series) -> dict | None:
    """Momentum + RSI stats for one symbol's close series (needs ~260 bars)."""
    close = close.dropna()
    if len(close) < 260:
        return None

    c_skip = close.iloc[-1 - SKIP]
    ret_63 = c_skip / close.iloc[-1 - SKIP - 63] - 1
    ret_126 = c_skip / close.iloc[-1 - SKIP - 126] - 1
    score = 0.5 * ret_63 + 0.5 * ret_126

    rsi = float(indicators.rsi(close, 14).iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])
    last = float(close.iloc[-1])

    if rsi < 30 and last > ma200:
        mr_flag = "oversold_dip"      # dip in an uptrend — the buyable kind
    elif rsi < 30:
        mr_flag = "oversold_downtrend"  # falling knife — regime filter decides
    elif rsi > 70:
        mr_flag = "overbought"
    else:
        mr_flag = ""

    return {
        "ret_63": float(ret_63),
        "ret_126": float(ret_126),
        "score": float(score),
        "rsi_14": rsi,
        "above_ma200": last > ma200,
        "mr_flag": mr_flag,
        "close": last,
    }


def rank_universe(closes: dict[str, pd.Series]) -> pd.DataFrame:
    """Build the cross-sectional table: one row per symbol, ranked by score.

    closes: {symbol: daily close series}. Symbols with insufficient history
    are dropped (logged), not errored — a robust basket tolerates gaps.
    """
    rows = {}
    for sym, close in closes.items():
        row = momentum_row(close)
        if row is None:
            logger.warning("rank_universe: skipping %s (insufficient history)", sym)
            continue
        rows[sym] = row

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "symbol"
    table = table.sort_values("score", ascending=False)
    table["rank"] = np.arange(1, len(table) + 1)
    return table


def momentum_longs(table: pd.DataFrame, top_n: int = 5, max_rsi: float = 75.0) -> list[str]:
    """Top-N momentum names, excluding already-overextended ones (RSI cap):
    buying a leader at RSI 80 is momentum-chasing, not momentum-following."""
    if table.empty:
        return []
    eligible = table[(table["rank"] <= top_n * 2) & (table["rsi_14"] <= max_rsi) & (table["score"] > 0)]
    return list(eligible.index[:top_n])


def momentum_shorts(table: pd.DataFrame, bottom_n: int = 3) -> list[str]:
    """Bottom-N momentum names that are ALSO below their 200d MA — the only
    shorts with both cross-sectional and absolute confirmation."""
    if table.empty:
        return []
    worst = table[(table["score"] < 0) & (~table["above_ma200"])]
    return list(worst.index[-bottom_n:]) if len(worst) else []


def dip_buys(table: pd.DataFrame, max_n: int = 3) -> list[str]:
    """Mean-reversion sleeve: oversold names still in their own uptrend."""
    if table.empty:
        return []
    dips = table[table["mr_flag"] == "oversold_dip"].sort_values("rsi_14")
    return list(dips.index[:max_n])
