"""Head-to-head model comparison on the same walk-forward pipeline.

Two phases:
  1. Per-symbol on a small basket (SPY, AAPL, BTC-USD) — quick directional read
  2. Class-pooled on equities (the best-test-bed class for ML) — power test

Same features, same WF cadence, same costs, same seed. Whoever wins wins fairly.

Run with:
    python compare_models.py
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import sources
import indicators
import regime
import backtest
import ml
import models
import class_xgb

logging.basicConfig(level=logging.WARNING)
np.random.seed(42)


def _market_ctx(start: str):
    try:
        from datetime import datetime, timezone
        from data import macro as macro_src
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return macro_src.get_market_context(start, end)
    except Exception as e:
        print(f"  (market context unavailable: {e})")
        return None


def prepare_for_ml(symbol: str, start: str, horizon: int = 10) -> pd.DataFrame:
    """Load OHLCV, attach indicators+regime, build features (incl. macro/SPY
    context — without it 6 of the 24 features are dead constants and the
    comparison can't see them)."""
    df = sources.get_bars(symbol, start, use_cache=True)
    df = indicators.attach_all(df)
    df = regime.label(df)
    feats = ml.make_features(df, horizon=horizon, macro_df=_market_ctx(start))
    return df, feats


def run_per_symbol(
    feats: pd.DataFrame,
    df: pd.DataFrame,
    train_size: int = 500,
    step_size: int = 60,
) -> dict[str, dict]:
    """Run every model on a single-symbol feature frame, backtest, return metrics."""
    feat_cols = list(ml.FEATURE_COLS)
    out: dict[str, dict] = {}
    for name, fit_fn in models.MODELS.items():
        t0 = time.time()
        try:
            proba = models.walk_forward_predict(
                feats, fit_fn, feature_cols=feat_cols,
                train_size=train_size, step_size=step_size,
                purge=10,  # horizon-length purge — without it every window trains on the test window's opening trend
            )
            positions = ml.proba_to_positions(proba)
            positions = positions.reindex(df.index, fill_value=0.0)
            r = backtest.run(df, positions, fee_bps=1.0, slippage_bps=1.0)
            m = r.metrics
            m["fit_seconds"] = round(time.time() - t0, 1)
            out[name] = m
        except Exception as e:
            out[name] = {"error": str(e), "fit_seconds": round(time.time() - t0, 1)}
    return out


def run_pooled_class(
    class_name: str,
    start: str,
    horizon: int = 10,
    train_size: int = 750,
    step_size: int = 60,
) -> dict[str, dict[str, dict]]:
    """For each model: pooled walk-forward on whole class, then per-symbol backtest."""
    symbols = class_xgb.CLASSES[class_name]
    symbol_dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = sources.get_bars(sym, start, use_cache=True)
            df = indicators.attach_all(df)
            df = regime.label(df)
            symbol_dfs[sym] = df
        except Exception as e:
            print(f"  skip {sym}: {e}")

    if not symbol_dfs:
        return {}

    pooled = class_xgb.build_pooled_features(symbol_dfs, horizon=horizon, macro_df=_market_ctx(start))
    feat_cols = list(ml.FEATURE_COLS)

    # Normalize date types for pooled walk-forward (matches class_xgb logic)
    pooled = pooled.copy()
    pooled["date"] = pd.to_datetime(pooled["date"], utc=True)
    pooled["forward_date"] = pd.to_datetime(pooled["forward_date"], utc=True)

    all_dates = sorted(pooled["date"].dropna().unique())
    n_dates = len(all_dates)

    results: dict[str, dict[str, dict]] = {}

    for model_name, fit_fn in models.MODELS.items():
        t0 = time.time()
        # Per-symbol probability series
        sym_probas: dict[str, pd.Series] = {}
        for sym in symbol_dfs:
            sym_dates = pd.DatetimeIndex(sorted(pooled.loc[pooled["symbol"] == sym, "date"].unique()))
            sym_probas[sym] = pd.Series(np.nan, index=sym_dates, dtype=float)

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

            try:
                X_train = train[feat_cols].astype(float).to_numpy()
                y_train = train["target"].astype(int).to_numpy()
                if len(np.unique(y_train)) < 2:
                    i += step_size
                    continue
                model = fit_fn(X_train, y_train)

                test = pooled[pooled["date"].isin(test_dates)].dropna(subset=feat_cols)
                if not test.empty:
                    proba = model.predict_proba(test[feat_cols].astype(float).to_numpy())[:, 1]
                    for ridx, p in zip(test.index, proba):
                        row = test.loc[ridx]
                        sym_probas[row["symbol"]].loc[row["date"]] = p
            except Exception as e:
                logging.debug("%s pooled step at %s failed: %s", model_name, cutoff, e)

            i += step_size

        elapsed = time.time() - t0

        # Backtest each symbol
        per_sym: dict[str, dict] = {}
        for sym, df in symbol_dfs.items():
            proba = sym_probas[sym]
            proba_aligned = proba.reindex(df.index)
            positions = pd.Series(0.0, index=df.index)
            positions[proba_aligned > 0.55] = 1.0
            positions[proba_aligned < 0.45] = -1.0
            try:
                m = backtest.run(df, positions, fee_bps=1.0, slippage_bps=1.0).metrics
            except Exception as e:
                m = {"error": str(e)}
            per_sym[sym] = m
        results[model_name] = {"_fit_seconds": round(elapsed, 1), **per_sym}

    return results


def fmt(v, fmt_str):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return fmt_str.format(v)


def print_per_symbol_table(symbol: str, results: dict[str, dict]):
    print(f"\n  {symbol}")
    print(f"  {'model':<14} {'win':>6} {'trades':>7} {'sharpe':>7} {'ret':>8} {'maxDD':>7} {'fit_s':>6}")
    print(f"  {'-'*14} {'-'*6} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*6}")
    for name, m in results.items():
        if "error" in m:
            print(f"  {name:<14} ERROR: {m['error'][:60]}")
            continue
        print(
            f"  {name:<14} {fmt(m['win_rate']*100, '{:>5.1f}%'):>6} "
            f"{int(m['trades']):>7d} {m['sharpe']:>+6.2f}  {fmt(m['total_return']*100, '{:>+5.1f}%'):>7} "
            f"{fmt(m['max_dd']*100, '{:>5.1f}%'):>7} {m.get('fit_seconds', 0):>6.1f}"
        )


def print_pooled_table(class_name: str, pooled_results: dict[str, dict[str, dict]]):
    print(f"\n  CLASS-POOLED: {class_name}")
    print(f"  {'model':<14} {'med_win':>8} {'med_ret':>8} {'med_sh':>7} {'%profit':>8} {'fit_s':>6}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*6}")

    for model_name, by_sym in pooled_results.items():
        sym_metrics = {k: v for k, v in by_sym.items() if not k.startswith("_") and "error" not in v}
        if not sym_metrics:
            print(f"  {model_name:<14} (no valid backtests)")
            continue
        wins = [m["win_rate"] for m in sym_metrics.values()]
        rets = [m["total_return"] for m in sym_metrics.values()]
        shs = [m["sharpe"] for m in sym_metrics.values()]
        n_profitable = sum(1 for r in rets if r > 0)
        n_total = len(rets)
        print(
            f"  {model_name:<14} {np.median(wins)*100:>7.1f}% "
            f"{np.median(rets)*100:>+7.1f}% {np.median(shs):>+6.2f} "
            f"{n_profitable}/{n_total:<2d}    {by_sym.get('_fit_seconds', 0):>6.1f}"
        )


def main() -> int:
    start_5y = (date.today() - timedelta(days=5 * 365)).isoformat()
    start_4y = (date.today() - timedelta(days=4 * 365)).isoformat()

    print("=" * 78)
    print("  MODEL COMPARISON — same WF, same features, same costs")
    print("=" * 78)

    # Phase 1: per-symbol on representative tickers
    print("\n[PHASE 1/2] Per-symbol walk-forward (5y daily, train=500, step=60)")
    for sym in ["SPY", "AAPL", "BTC-USD"]:
        try:
            df, feats = prepare_for_ml(sym, start_5y)
        except Exception as e:
            print(f"  {sym} prep failed: {e}")
            continue
        results = run_per_symbol(feats, df)
        print_per_symbol_table(sym, results)

    # Phase 2: class-pooled on equities (the strongest test bed)
    print("\n[PHASE 2/2] Class-pooled walk-forward (4y, equities x 7, train=750, step=60)")
    pooled = run_pooled_class("equities", start_4y)
    print_pooled_table("equities", pooled)

    print("\n" + "=" * 78)
    print("""  HOW TO READ THIS

  Per-symbol section: which model wins on which ticker, all else equal.
  Pooled section: which model best exploits ~5,000 pooled training rows.

  The clear winner is the model that:
    - has highest median win rate AND
    - has positive median sharpe AND
    - is profitable on the most symbols
""")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
