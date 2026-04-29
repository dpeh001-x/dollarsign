"""Stock screener — scan a wide universe of tickers for actionable signals.

The expanded universe (~85 symbols) is split into ad-hoc pools:
  * equities (~46 large/mega-caps): one pooled model serves all
  * broad_etf (7): one pooled model serves all
  * crypto (3): one pooled model serves all
  * sector_etf (11), industry_etf (6), macro_etf (8), intl_etf (6):
    per-symbol training (these classes are too heterogeneous to pool well)

Pooling reduces model fits dramatically — 56 of 85 symbols are covered
by just 3 fits. The other 29 train per-symbol. Total ~32 model fits for
85 symbols, all cached by the underlying class_xgb / ml infrastructure
for 2h.

The screener does NOT apply a confidence filter — returns ALL symbols
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


# ─── Expanded universe (~85 symbols, pooled where it pays off) ──────────
EQUITY_POOL: list[str] = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL",
    "CRM", "ADBE", "AMD", "INTC", "CSCO", "QCOM", "TXN", "IBM",
    # Financial
    "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP", "BLK", "SCHW",
    # Healthcare
    "LLY", "JNJ", "UNH", "PFE", "MRK", "ABBV", "TMO", "DHR", "BMY",
    # Consumer / retail
    "WMT", "HD", "COST", "MCD", "NKE", "DIS", "SBUX", "TGT", "LOW",
    # Industrials / energy / materials
    "CAT", "DE", "GE", "BA", "HON", "XOM", "CVX", "COP",
    # Telecom / staples
    "T", "VZ", "KO", "PEP", "PG", "PM", "MO",
    # Networking / SaaS
    "NOW", "SNOW", "PLTR", "UBER", "ABNB",
]

BROAD_ETF_POOL: list[str] = ["SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "RSP"]

CRYPTO_POOL: list[str] = ["BTC-USD", "ETH-USD", "SOL-USD"]

SECTOR_ETFS: list[str] = [
    "XLF", "XLE", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU", "XLRE", "XLC", "XLB",
]

INDUSTRY_ETFS: list[str] = ["SMH", "SOXX", "XBI", "KRE", "ITB", "KWEB"]

MACRO_ETFS: list[str] = ["TLT", "IEF", "BIL", "GLD", "SLV", "USO", "DBC", "DBA"]

INTL_ETFS: list[str] = ["EFA", "EEM", "FXI", "EWJ", "EWG", "INDA"]

# For backwards compat / CLI summary
SCREEN_UNIVERSE_SIZE = (
    len(EQUITY_POOL) + len(BROAD_ETF_POOL) + len(CRYPTO_POOL)
    + len(SECTOR_ETFS) + len(INDUSTRY_ETFS) + len(MACRO_ETFS) + len(INTL_ETFS)
)


@dataclass
class ScreenResult:
    ticker: str
    asset_class: str
    signal: str          # "long" | "short" | "flat"
    proba: float
    confidence: float
    last_close: float


# ─── Pool prediction: trains one model on N symbols' pooled features ────
def _pool_predict(
    pool_label: str,
    symbols: list[str],
    start_iso: str,
    horizon: int = 10,
    use_ensemble: bool = True,
    long_thresh: float = 0.575,
    short_thresh: float = 0.425,
) -> list[ScreenResult]:
    """Train ONE model on pooled features from `symbols`, predict latest bar for each."""
    import models as _models  # for ensemble fit functions

    symbol_dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            symbol_dfs[sym] = class_xgb.prepare_symbol(sym, start_iso)
        except Exception as e:
            logger.warning("prep %s: %s", sym, e)
    if not symbol_dfs:
        return []

    pooled = class_xgb.build_pooled_features(symbol_dfs, horizon=horizon)
    feat_cols = list(ml.FEATURE_COLS)
    train = pooled.dropna(subset=["target"]).dropna(subset=feat_cols)
    if len(train) < 200:
        logger.warning("Not enough pooled training data for %s: %d rows", pool_label, len(train))
        return []

    X = train[feat_cols].astype(float).to_numpy()
    y = train["target"].astype(int).to_numpy()

    fitted_models: list = []
    if use_ensemble:
        for name in ml.ENSEMBLE_MODELS:
            fit_fn = _models.MODELS.get(name)
            if fit_fn is None:
                continue
            try:
                fitted_models.append(fit_fn(X, y))
            except Exception as e:
                logger.warning("ensemble fit %s on %s: %s", name, pool_label, e)
    else:
        try:
            fitted_models = [ml.fit_xgb(train)]
        except Exception as e:
            logger.warning("fit_xgb on %s: %s", pool_label, e)
            return []

    if not fitted_models:
        return []

    out: list[ScreenResult] = []
    for sym, df in symbol_dfs.items():
        sym_rows = pooled[pooled["symbol"] == sym].dropna(subset=feat_cols)
        if sym_rows.empty:
            continue
        latest = sym_rows.iloc[-1]
        Xrow = latest[feat_cols].astype(float).to_numpy().reshape(1, -1)

        probas = []
        for m in fitted_models:
            try:
                probas.append(float(m.predict_proba(Xrow)[0, 1]))
            except Exception:
                pass
        if not probas:
            continue
        proba = sum(probas) / len(probas)

        if proba > long_thresh:
            signal = "long"
        elif proba < short_thresh:
            signal = "short"
        else:
            signal = "flat"

        out.append(ScreenResult(
            ticker=sym, asset_class=pool_label,
            signal=signal, proba=proba,
            confidence=abs(proba - 0.5) * 2,
            last_close=float(df["close"].iloc[-1]),
        ))

    return out


def _per_symbol_predict(
    pool_label: str,
    symbols: list[str],
    start_iso: str,
    horizon: int = 10,
    use_ensemble: bool = True,
) -> list[ScreenResult]:
    """For each symbol, train its own model. Used for heterogeneous classes."""
    out: list[ScreenResult] = []
    for sym in symbols:
        try:
            df = sources.get_bars(sym, start_iso, use_cache=True)
            df = indicators.attach_all(df)
            df = regime.label(df)
            if use_ensemble:
                predictor = ml.EnsemblePredictor(horizon=horizon).fit(df)
            else:
                predictor = ml.XGBPredictor(horizon=horizon).fit(df)
            sig = predictor.predict_now(df)
            out.append(ScreenResult(
                ticker=sym, asset_class=pool_label,
                signal=sig["signal"], proba=float(sig["proba"]),
                confidence=float(sig["confidence"]),
                last_close=float(df["close"].iloc[-1]),
            ))
        except Exception as e:
            logger.warning("per-symbol %s: %s", sym, e)
    return out


def screen(use_ensemble: bool = True, horizon: int = 10) -> list[ScreenResult]:
    """Full universe scan. Returns sorted by confidence (highest first)."""
    start_iso = (date.today() - timedelta(days=3 * 365)).isoformat()
    results: list[ScreenResult] = []

    # Pooled training (one model per pool, predicts for all symbols in pool)
    pooled_groups = [
        ("equities", EQUITY_POOL),
        ("broad_etf", BROAD_ETF_POOL),
        ("crypto", CRYPTO_POOL),
    ]
    for label, syms in pooled_groups:
        results.extend(_pool_predict(label, syms, start_iso, horizon=horizon, use_ensemble=use_ensemble))

    # Per-symbol training (heterogeneous classes)
    per_symbol_groups = [
        ("sector_etf", SECTOR_ETFS),
        ("industry_etf", INDUSTRY_ETFS),
        ("macro_etf", MACRO_ETFS),
        ("intl_etf", INTL_ETFS),
    ]
    for label, syms in per_symbol_groups:
        results.extend(_per_symbol_predict(label, syms, start_iso, horizon=horizon, use_ensemble=use_ensemble))

    results.sort(key=lambda r: r.confidence, reverse=True)
    return results


def filter_actionable(results: list[ScreenResult], min_confidence: float = 0.75) -> list[ScreenResult]:
    return [r for r in results if r.confidence >= min_confidence and r.signal != "flat"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-conf", type=float, default=0.75)
    parser.add_argument("--no-ensemble", action="store_true")
    args = parser.parse_args()

    print(f"\nScreening {SCREEN_UNIVERSE_SIZE} symbols (ensemble={'no' if args.no_ensemble else 'yes'}, min-conf={args.min_conf:.0%})...\n")
    rows = screen(use_ensemble=not args.no_ensemble)
    actionable = filter_actionable(rows, min_confidence=args.min_conf)

    print(f"{'#':>4} {'ticker':<10} {'class':<13} {'signal':<5} {'P(up)':>7} {'conf':>6} {'price':>10}")
    print(f"{'-'*4} {'-'*10} {'-'*13} {'-'*5} {'-'*7} {'-'*6} {'-'*10}")
    for i, r in enumerate(rows, 1):
        marker = " ★ " if r in actionable else "   "
        print(f"{i:>3}{marker}{r.ticker:<10} {r.asset_class:<13} {r.signal[:4]:<5} {r.proba*100:>6.1f}% {r.confidence*100:>5.0f}% {r.last_close:>10.2f}")
    print(f"\nActionable (conf >= {args.min_conf:.0%}): {len(actionable)} of {len(rows)}")
