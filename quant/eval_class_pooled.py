"""Side-by-side eval: per-symbol XGB vs class-pooled XGB.

For each asset class:
  1. Train one pooled model with leakage-controlled walk-forward
  2. For each symbol in class, also run the existing per-symbol XGB
  3. Backtest both and compare metrics
Output: per-class table, then overall medians + count of profitable symbols.
"""
from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backtest
import class_xgb
import strategies

logging.basicConfig(level=logging.WARNING)


def main() -> int:
    start = (date.today() - timedelta(days=4 * 365)).isoformat()

    print(f"\n{'='*82}")
    print(f"  PER-SYMBOL XGB  vs  CLASS-POOLED XGB     (4y, fees+slip = 2 bps/side)")
    print(f"{'='*82}")

    rows: list[dict] = []

    for cls_name, symbols in class_xgb.CLASSES.items():
        print(f"\n[{cls_name}] training pooled model on {len(symbols)} symbols...")
        try:
            class_probas = class_xgb.predict_class(
                cls_name, start, horizon=10, train_size=750, step_size=60
            )
        except Exception as e:
            print(f"  failed: {e}")
            continue

        print(f"  {'symbol':<10} {'per_win':>8} {'per_n':>5} {'per_ret':>8} {'per_sh':>7}  | "
              f"{'pool_win':>9} {'pool_n':>6} {'pool_ret':>9} {'pool_sh':>8}  | {'Δwin':>6} {'Δret':>7}")
        print(f"  {'-'*10} {'-'*8} {'-'*5} {'-'*8} {'-'*7}--+-{'-'*9} {'-'*6} {'-'*9} {'-'*8}--+-{'-'*6} {'-'*7}")

        for sym in symbols:
            try:
                df = class_xgb.prepare_symbol(sym, start)
            except Exception as e:
                print(f"  {sym:<10} (prep failed: {e})")
                continue

            # Per-symbol XGB (existing strategy)
            try:
                pos_per = strategies.xgb_walk_forward(df)
                m_per = backtest.run(df, pos_per, fee_bps=1.0, slippage_bps=1.0).metrics
            except Exception as e:
                print(f"  {sym:<10} (per-symbol failed: {e})")
                continue

            # Class-pooled XGB
            try:
                proba = class_probas.get(sym)
                if proba is None or proba.dropna().empty:
                    raise RuntimeError("no pooled predictions")
                proba_aligned = proba.reindex(df.index)
                pos_pool = pd.Series(0.0, index=df.index)
                pos_pool[proba_aligned > 0.55] = 1.0
                pos_pool[proba_aligned < 0.45] = -1.0
                m_pool = backtest.run(df, pos_pool, fee_bps=1.0, slippage_bps=1.0).metrics
            except Exception as e:
                print(f"  {sym:<10} (pooled failed: {e})")
                m_pool = {"trades": 0, "win_rate": np.nan, "total_return": np.nan, "sharpe": np.nan, "max_dd": np.nan}

            rows.append({
                "class": cls_name,
                "symbol": sym,
                "per_trades": int(m_per["trades"]),
                "per_win": m_per["win_rate"],
                "per_return": m_per["total_return"],
                "per_sharpe": m_per["sharpe"],
                "pool_trades": int(m_pool["trades"]),
                "pool_win": m_pool["win_rate"],
                "pool_return": m_pool["total_return"],
                "pool_sharpe": m_pool["sharpe"],
            })

            d_win = (m_pool["win_rate"] - m_per["win_rate"]) * 100 if not np.isnan(m_pool["win_rate"]) else np.nan
            d_ret = (m_pool["total_return"] - m_per["total_return"]) * 100 if not np.isnan(m_pool["total_return"]) else np.nan
            print(
                f"  {sym:<10} {m_per['win_rate']*100:>7.1f}% {m_per['trades']:>5.0f} "
                f"{m_per['total_return']*100:>+7.1f}% {m_per['sharpe']:>+6.2f}  | "
                f"{m_pool['win_rate']*100:>8.1f}% {m_pool['trades']:>6.0f} "
                f"{m_pool['total_return']*100:>+8.1f}% {m_pool['sharpe']:>+7.2f}  | "
                f"{d_win:>+5.1f} {d_ret:>+6.1f}"
            )

    if not rows:
        print("\nNo results.")
        return 1

    df_results = pd.DataFrame(rows)

    print(f"\n{'='*82}")
    print(f"  AGGREGATED COMPARISON")
    print(f"{'='*82}")

    print("\n  Per-class summary (median across symbols within class):")
    print(f"  {'class':<14}  {'per_win':>8} {'per_ret':>8} {'per_sh':>7} | "
          f"{'pool_win':>9} {'pool_ret':>9} {'pool_sh':>8} | {'%per_prof':>10} {'%pool_prof':>11}")
    for cls in df_results["class"].unique():
        sub = df_results[df_results["class"] == cls]
        per_w = sub["per_win"].median() * 100
        per_r = sub["per_return"].median() * 100
        per_s = sub["per_sharpe"].median()
        pool_w = sub["pool_win"].median() * 100
        pool_r = sub["pool_return"].median() * 100
        pool_s = sub["pool_sharpe"].median()
        per_pf = (sub["per_return"] > 0).mean() * 100
        pool_pf = (sub["pool_return"] > 0).mean() * 100
        print(
            f"  {cls:<14}  {per_w:>7.1f}% {per_r:>+7.1f}% {per_s:>+6.2f} | "
            f"{pool_w:>8.1f}% {pool_r:>+8.1f}% {pool_s:>+7.2f} | "
            f"{per_pf:>9.0f}% {pool_pf:>10.0f}%"
        )

    print(f"\n  Overall (median across all {len(df_results)} symbols):")
    per_w_all = df_results["per_win"].median() * 100
    per_r_all = df_results["per_return"].median() * 100
    per_s_all = df_results["per_sharpe"].median()
    pool_w_all = df_results["pool_win"].median() * 100
    pool_r_all = df_results["pool_return"].median() * 100
    pool_s_all = df_results["pool_sharpe"].median()

    n_per_prof = (df_results["per_return"] > 0).sum()
    n_pool_prof = (df_results["pool_return"] > 0).sum()
    n_total = len(df_results)

    print(f"  per-symbol XGB:  win={per_w_all:.1f}%  return={per_r_all:+.1f}%  "
          f"sharpe={per_s_all:+.2f}  profitable={n_per_prof}/{n_total} ({n_per_prof/n_total*100:.0f}%)")
    print(f"  class-pooled:    win={pool_w_all:.1f}%  return={pool_r_all:+.1f}%  "
          f"sharpe={pool_s_all:+.2f}  profitable={n_pool_prof}/{n_total} ({n_pool_prof/n_total*100:.0f}%)")
    print(f"  delta (pool - per): win {pool_w_all - per_w_all:+.1f}pp  "
          f"return {pool_r_all - per_r_all:+.1f}pp  sharpe {pool_s_all - per_s_all:+.2f}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
