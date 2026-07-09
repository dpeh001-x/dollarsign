"""Trend & momentum tracker — pure indicator scan, no ML.

Complements the ML screener: where the screener asks "what does the model
predict for the next 10 days?", this tracker asks "what is each symbol
DOING right now?" — sustained trend strength, momentum acceleration,
overbought/oversold. Useful for confirming model leans or spotting
fresh breakouts the model hasn't latched onto yet.

Trend score (-1..+1):
  * SMA50 vs SMA200 alignment           ±0.40
  * Price vs SMA200                     ±0.30
  * ADX strength × DI direction         ±0.30   (saturates above ADX=40)

Momentum score (-1..+1):
  * 20-day return (clipped at ±15%)     ±0.40
  * 5-day return  (clipped at ±8%)      ±0.25
  * RSI distance from 50                ±0.20   (RSI 20 -> -0.20, RSI 80 -> +0.20)
  * MACD histogram sign × magnitude     ±0.15

A "combined" score is the sum (trend + momentum) — aligned trend &
momentum surfaces the strongest directional setups.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import indicators
from data import sources
from screener import (
    EQUITY_POOL, BROAD_ETF_POOL, CRYPTO_POOL,
    SECTOR_ETFS, INDUSTRY_ETFS, MACRO_ETFS, INTL_ETFS,
    SCREEN_UNIVERSE_SIZE,
)

logger = logging.getLogger(__name__)


UNIVERSE: list[tuple[str, list[str]]] = [
    ("equities", EQUITY_POOL),
    ("broad_etf", BROAD_ETF_POOL),
    ("crypto", CRYPTO_POOL),
    ("sector_etf", SECTOR_ETFS),
    ("industry_etf", INDUSTRY_ETFS),
    ("macro_etf", MACRO_ETFS),
    ("intl_etf", INTL_ETFS),
]


@dataclass
class TrendMomentumResult:
    ticker: str
    asset_class: str
    last_close: float
    # Trend
    trend_score: float          # -1..+1
    trend_label: str            # "strong_up" | "weak_up" | "ranging" | "weak_down" | "strong_down"
    adx: float
    sma50: float
    sma200: float
    # Momentum
    momentum_score: float       # -1..+1
    momentum_label: str         # "accelerating_up" | "bullish" | "neutral" | "bearish" | "accelerating_down" | "overbought" | "oversold"
    rsi_14: float
    ret_5d: float               # pct
    ret_20d: float              # pct
    macd_hist: float
    # Combined
    combined_score: float       # trend + momentum, range -2..+2
    aligned: bool               # trend & momentum point the same way (and both nonzero)


def _trend_score(close: float, sma50: float, sma200: float,
                 adx: float, plus_di: float, minus_di: float) -> float:
    score = 0.0
    if not np.isnan(sma50) and not np.isnan(sma200):
        score += 0.40 if sma50 > sma200 else -0.40
        score += 0.30 if close > sma200 else -0.30
    if not np.isnan(adx) and not np.isnan(plus_di) and not np.isnan(minus_di):
        direction = 1.0 if plus_di > minus_di else -1.0
        strength = min(adx / 40.0, 1.0)
        score += direction * strength * 0.30
    return float(np.clip(score, -1.0, 1.0))


def _trend_label(score: float, adx: float) -> str:
    if score > 0.6:
        return "strong_up"
    if score > 0.25:
        return "weak_up"
    if score < -0.6:
        return "strong_down"
    if score < -0.25:
        return "weak_down"
    return "ranging"


def _momentum_score(rsi: float, ret_5d: float, ret_20d: float,
                    macd_hist: float, close: float) -> float:
    score = 0.0
    if not np.isnan(ret_20d):
        score += np.clip(ret_20d / 0.15, -1.0, 1.0) * 0.40
    if not np.isnan(ret_5d):
        score += np.clip(ret_5d / 0.08, -1.0, 1.0) * 0.25
    if not np.isnan(rsi):
        score += np.clip((rsi - 50) / 30.0, -1.0, 1.0) * 0.20
    if not np.isnan(macd_hist) and close > 0:
        score += np.clip((macd_hist / close) / 0.005, -1.0, 1.0) * 0.15
    return float(np.clip(score, -1.0, 1.0))


def _momentum_label(score: float, rsi: float) -> str:
    if not np.isnan(rsi):
        if rsi >= 75:
            return "overbought"
        if rsi <= 25:
            return "oversold"
    if score > 0.6:
        return "accelerating_up"
    if score > 0.2:
        return "bullish"
    if score < -0.6:
        return "accelerating_down"
    if score < -0.2:
        return "bearish"
    return "neutral"


def analyze_from_frame(df: pd.DataFrame, symbol: str = "", asset_class: str = "") -> TrendMomentumResult | None:
    """Compute trend + momentum scores from a frame that already has indicators
    attached (via indicators.attach_all). Avoids a redundant data fetch when the
    caller already has the symbol loaded."""
    if df is None or df.empty or len(df) < 210:
        return None
    close = float(df["close"].iloc[-1])
    sma50 = float(df["close"].rolling(50).mean().iloc[-1])
    sma200 = float(df["close"].rolling(200).mean().iloc[-1])
    adx = float(df["adx"].iloc[-1]) if "adx" in df.columns else np.nan
    plus_di = float(df["plus_di"].iloc[-1]) if "plus_di" in df.columns else np.nan
    minus_di = float(df["minus_di"].iloc[-1]) if "minus_di" in df.columns else np.nan
    rsi = float(df["rsi_14"].iloc[-1])
    ret_5d = float(df["close"].iloc[-1] / df["close"].iloc[-6] - 1) if len(df) > 6 else np.nan
    ret_20d = float(df["close"].iloc[-1] / df["close"].iloc[-21] - 1) if len(df) > 21 else np.nan
    macd_hist = float(df["macd_hist"].iloc[-1]) if "macd_hist" in df.columns else np.nan

    tscore = _trend_score(close, sma50, sma200, adx, plus_di, minus_di)
    mscore = _momentum_score(rsi, ret_5d, ret_20d, macd_hist, close)
    tlabel = _trend_label(tscore, adx)
    mlabel = _momentum_label(mscore, rsi)

    aligned = (tscore > 0.25 and mscore > 0.2) or (tscore < -0.25 and mscore < -0.2)

    return TrendMomentumResult(
        ticker=symbol, asset_class=asset_class, last_close=close,
        trend_score=tscore, trend_label=tlabel,
        adx=adx, sma50=sma50, sma200=sma200,
        momentum_score=mscore, momentum_label=mlabel,
        rsi_14=rsi, ret_5d=ret_5d, ret_20d=ret_20d, macd_hist=macd_hist,
        combined_score=tscore + mscore, aligned=aligned,
    )


def _analyze_symbol(symbol: str, asset_class: str, start_iso: str) -> TrendMomentumResult | None:
    try:
        df = sources.get_bars(symbol, start_iso, use_cache=True)
    except Exception as e:
        logger.warning("fetch %s: %s", symbol, e)
        return None
    if df is None or df.empty or len(df) < 210:
        return None
    df = indicators.attach_all(df)
    return analyze_from_frame(df, symbol=symbol, asset_class=asset_class)


def track() -> list[TrendMomentumResult]:
    """Scan the screener universe for trend + momentum readings.

    Returns sorted by |combined_score| desc (strongest setups first).
    """
    start_iso = (date.today() - timedelta(days=3 * 365)).isoformat()
    results: list[TrendMomentumResult] = []
    for asset_class, symbols in UNIVERSE:
        for sym in symbols:
            r = _analyze_symbol(sym, asset_class, start_iso)
            if r is not None:
                results.append(r)
    results.sort(key=lambda r: abs(r.combined_score), reverse=True)
    return results


if __name__ == "__main__":
    print(f"\nScanning {SCREEN_UNIVERSE_SIZE} symbols for trend + momentum...\n")
    rows = track()
    print(f"{'ticker':<10} {'class':<13} {'trend':<11} {'mom':<18} {'tS':>6} {'mS':>6} {'cS':>6} {'aln':>4} {'rsi':>5} {'adx':>5} {'5d%':>7} {'20d%':>7}")
    print("-" * 110)
    for r in rows:
        print(f"{r.ticker:<10} {r.asset_class:<13} {r.trend_label:<11} {r.momentum_label:<18} "
              f"{r.trend_score:>+6.2f} {r.momentum_score:>+6.2f} {r.combined_score:>+6.2f} "
              f"{'Y' if r.aligned else '-':>4} {r.rsi_14:>5.0f} {r.adx:>5.0f} "
              f"{r.ret_5d*100:>+6.1f}% {r.ret_20d*100:>+6.1f}%")
    aligned = [r for r in rows if r.aligned]
    print(f"\nAligned setups (trend & momentum agree): {len(aligned)} of {len(rows)}")
