"""Stock screener — scan a universe of liquid tickers for actionable signals.

For each symbol in the screening universe, computes the current XGBoost
prediction (using class-pooled training where beneficial, per-symbol
otherwise — same logic as the main app). Returns ranked list with
direction, P(up), confidence, and last close.

Pool-class symbols (equities, broad ETFs, crypto) share their model — only
3 model fits cover 13 symbols. Per-symbol classes (sector ETFs, macro
ETFs) need 1 fit each. Total: ~15 model fits for 25 symbols, all cached
by the underlying class_xgb / ml infrastructure.

The screener does NOT apply a confidence filter — it returns ALL symbols
with their raw probabilities, so callers can apply their own threshold.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import indicators
import regime
import ml
import class_xgb
from data import sources

logger = logging.getLogger(__name__)


# Default 25-symbol screening universe — liquid, broadly representative
SCREEN_UNIVERSE: list[tuple[str, str]] = [
    # Mega-cap equities (pooled)
    ("AAPL", "equities"), ("MSFT", "equities"), ("GOOGL", "equities"),
    ("AMZN", "equities"), ("META", "equities"), ("NVDA", "equities"),
    ("TSLA", "equities"),
    # Broad ETFs (pooled)
    ("SPY", "broad_etf"), ("QQQ", "broad_etf"),
    ("IWM", "broad_etf"), ("DIA", "broad_etf"),
    # Crypto (pooled)
    ("BTC-USD", "crypto"), ("ETH-USD", "crypto"),
    # Sector ETFs (per-symbol)
    ("XLF", "sector_etf"), ("XLE", "sector_etf"), ("XLK", "sector_etf"),
    ("XLV", "sector_etf"), ("XLY", "sector_etf"),
    ("XLP", "sector_etf"), ("XLI", "sector_etf"),
    # Macro / commodity ETFs (per-symbol)
    ("TLT", "macro_etf"), ("GLD", "macro_etf"), ("USO", "macro_etf"),
    ("EFA", "macro_etf"), ("EEM", "macro_etf"),
]


@dataclass
class ScreenResult:
    ticker: str
    asset_class: str
    signal: str          # "long" | "short" | "flat"
    proba: float         # P(up) in [0,1]
    confidence: float    # |proba - 0.5| * 2
    last_close: float


def screen(use_ensemble: bool = True, horizon: int = 10) -> list[ScreenResult]:
    """Run the screen across the default universe. Returns sorted by confidence (highest first)."""
    start_iso = (date.today() - timedelta(days=3 * 365)).isoformat()

    # Group by class for efficient pooled training
    pooled_classes: dict[str, list[str]] = {"equities": [], "broad_etf": [], "crypto": []}
    per_symbol: list[tuple[str, str]] = []
    for sym, cls in SCREEN_UNIVERSE:
        if cls in class_xgb.POOL_BENEFITS:
            pooled_classes[cls].append(sym)
        else:
            per_symbol.append((sym, cls))

    results: list[ScreenResult] = []

    # Pooled — one model per class produces predictions for all symbols in class
    for cls, syms in pooled_classes.items():
        if not syms:
            continue
        try:
            if use_ensemble:
                pool_pred = class_xgb.predict_today_for_class_ensemble(cls, start_iso, horizon=horizon)
            else:
                pool_pred = class_xgb.predict_today_for_class(cls, start_iso, horizon=horizon)
        except Exception as e:
            logger.warning("Pool predict failed for %s: %s", cls, e)
            continue

        for sym in syms:
            sig = pool_pred.get(sym)
            if sig is None:
                continue
            try:
                df = sources.get_bars(sym, start_iso, use_cache=True)
                last_close = float(df["close"].iloc[-1])
            except Exception:
                last_close = float("nan")
            results.append(ScreenResult(
                ticker=sym, asset_class=cls,
                signal=sig["signal"], proba=float(sig["proba"]),
                confidence=float(sig["confidence"]), last_close=last_close,
            ))

    # Per-symbol — one model fit per symbol
    for sym, cls in per_symbol:
        try:
            df = sources.get_bars(sym, start_iso, use_cache=True)
            df = indicators.attach_all(df)
            df = regime.label(df)
            if use_ensemble:
                predictor = ml.EnsemblePredictor(horizon=horizon).fit(df)
            else:
                predictor = ml.XGBPredictor(horizon=horizon).fit(df)
            sig = predictor.predict_now(df)
        except Exception as e:
            logger.warning("Per-symbol predict failed for %s: %s", sym, e)
            continue

        results.append(ScreenResult(
            ticker=sym, asset_class=cls,
            signal=sig["signal"], proba=float(sig["proba"]),
            confidence=float(sig["confidence"]),
            last_close=float(df["close"].iloc[-1]),
        ))

    # Sort by confidence descending — highest-conviction first
    results.sort(key=lambda r: r.confidence, reverse=True)
    return results


def filter_actionable(results: list[ScreenResult], min_confidence: float = 0.75) -> list[ScreenResult]:
    """Filter to symbols meeting the minimum confidence threshold."""
    return [r for r in results if r.confidence >= min_confidence and r.signal != "flat"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-conf", type=float, default=0.75)
    parser.add_argument("--no-ensemble", action="store_true")
    args = parser.parse_args()

    print(f"\nScreening {len(SCREEN_UNIVERSE)} symbols (ensemble={'no' if args.no_ensemble else 'yes'}, min-conf={args.min_conf:.0%})...\n")
    rows = screen(use_ensemble=not args.no_ensemble)
    actionable = filter_actionable(rows, min_confidence=args.min_conf)

    print(f"{'#':>3} {'ticker':<10} {'class':<11} {'signal':<5} {'P(up)':>7} {'conf':>6} {'price':>10}")
    print(f"{'-'*3} {'-'*10} {'-'*11} {'-'*5} {'-'*7} {'-'*6} {'-'*10}")
    for i, r in enumerate(rows, 1):
        marker = " * " if r in actionable else "   "
        print(f"{marker}{r.ticker:<10} {r.asset_class:<11} {r.signal[:4]:<5} {r.proba*100:>6.1f}% {r.confidence*100:>5.0f}% {r.last_close:>10.2f}")
    print(f"\nActionable (conf >= {args.min_conf:.0%}): {len(actionable)} of {len(rows)}")
