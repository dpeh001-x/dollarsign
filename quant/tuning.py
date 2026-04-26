"""Parameter grid-search with selection-bias controls.

Why this exists: if you grid-search 100 parameter combinations and pick
the one with the highest Sharpe on test data, that "best" result is
contaminated by selection bias — you've effectively run 100 hypothesis
tests and reported the most extreme one. The naive p-value is meaningless.

This module enforces three controls:

  1. Time-respecting train/test split — no random shuffling. Train on the
     first 60% of bars, test on the held-out 40%.
  2. Train-set parameter selection — pick the param combo that maximizes
     Sharpe on TRAIN. Then report its TEST performance honestly. (Hyper-
     parameters chosen with knowledge of the test set are cheating.)
  3. Benjamini-Hochberg FDR correction — for each combo, compute a binomial
     two-sided p-value against the 50%-win-rate null on its test trades.
     BH correction across ALL combos controls the false-discovery rate;
     a combo is "robustly significant" only if it survives BH at FDR=0.10.

The script also reports the deflation between train and test Sharpe per
strategy. Large positive deflation = the strategy is a chameleon, fitting
noise in train and falling apart out-of-sample.

Run with:
    python tuning.py                 # tune all strategies on SPY
    python tuning.py --symbol QQQ    # tune on a different symbol
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import sources
import indicators
import regime
import backtest
import strategies

logging.basicConfig(level=logging.WARNING)


# ─── Parameter grids ───────────────────────────────────────────────────
GRIDS: dict[str, dict[str, list[Any]]] = {
    "regime_breakout": {
        "lookback": [10, 20, 30, 50],
        "rsi_overbought": [60, 65, 70, 75, 80],
        "rsi_oversold": [20, 25, 30, 35, 40],
    },
    "rsi_meanrev": {
        "oversold": [20, 25, 30, 35],
        "overbought": [65, 70, 75, 80],
    },
    "ma_cross": {
        "fast": [5, 10, 20, 30, 50],
        "slow": [50, 100, 150, 200],
    },
}


def grid_combos(grid: dict[str, list[Any]]):
    keys = list(grid.keys())
    for combo in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, combo))


def is_valid(strat: str, params: dict) -> bool:
    if strat == "ma_cross":
        return params["fast"] < params["slow"]
    if strat == "rsi_meanrev":
        return params["oversold"] < params["overbought"]
    if strat == "regime_breakout":
        return params["rsi_oversold"] < params["rsi_overbought"]
    return True


def positions_for(strat: str, df: pd.DataFrame, params: dict) -> pd.Series:
    if strat == "regime_breakout":
        return strategies.regime_aware_breakout(df, **params)
    if strat == "rsi_meanrev":
        return strategies.rsi_mean_reversion(df, **params)
    if strat == "ma_cross":
        return strategies.ma_crossover(df, **params)
    raise KeyError(strat)


# ─── Time-respecting split ─────────────────────────────────────────────
def time_split(df: pd.DataFrame, train_frac: float = 0.6):
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


# ─── Statistical tests ─────────────────────────────────────────────────
def winrate_pvalue(wins: int, total: int) -> float:
    """Two-sided binomial test against null p=0.5."""
    if total == 0:
        return 1.0
    return stats.binomtest(wins, total, p=0.5, alternative="two-sided").pvalue


def benjamini_hochberg(pvalues: np.ndarray, fdr: float = 0.10) -> np.ndarray:
    """Step-up procedure. Returns boolean mask of significant tests at given FDR.

    Find the largest k where p_(k) <= (k/m) * fdr; reject H0 for all
    p-values <= p_(k). Controls expected false discovery rate.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    threshold = (np.arange(1, m + 1) / m) * fdr
    crossed = sorted_p <= threshold
    if not crossed.any():
        return np.zeros(m, dtype=bool)
    k_max = np.where(crossed)[0].max()
    sig = np.zeros(m, dtype=bool)
    sig[order[: k_max + 1]] = True
    return sig


# ─── Grid search per strategy ──────────────────────────────────────────
@dataclass
class TuningResult:
    strategy: str
    grid_size: int
    train_best_params: dict
    test_metrics_at_train_best: dict
    cherry_pick_test_best: dict           # most "impressive" test result (selection-bias warning)
    cherry_pick_test_best_params: dict
    train_test_sharpe_deflation: float    # mean(train_sharpe - test_sharpe) across grid
    n_significant_bh10: int               # combos that survive BH at FDR 10%
    n_significant_bh05: int


def tune_strategy(
    full_df: pd.DataFrame,
    strat: str,
    grid: dict[str, list[Any]],
    train_frac: float = 0.6,
) -> tuple[pd.DataFrame, TuningResult]:
    train_df, test_df = time_split(full_df, train_frac)

    # Refit regime classifier on TRAIN only — using full-sample fit would
    # leak future info into the train labels.
    regime_model = regime.fit(train_df)
    train_df["regime"] = regime_model.predict(train_df)
    test_df["regime"] = regime_model.predict(test_df)

    rows = []
    for params in grid_combos(grid):
        if not is_valid(strat, params):
            continue
        try:
            train_pos = positions_for(strat, train_df, params)
            test_pos = positions_for(strat, test_df, params)
            train_r = backtest.run(train_df, train_pos, fee_bps=1.0, slippage_bps=1.0)
            test_r = backtest.run(test_df, test_pos, fee_bps=1.0, slippage_bps=1.0)
        except Exception:
            continue

        train_m, test_m = train_r.metrics, test_r.metrics

        if test_m["trades"] > 0 and len(test_r.trades) > 0:
            wins = int((test_r.trades["net_pct"] > 0).sum())
            total = int(test_m["trades"])
            p_value = winrate_pvalue(wins, total)
        else:
            wins, total, p_value = 0, 0, 1.0

        rows.append({
            **params,
            "train_sharpe": train_m["sharpe"],
            "train_winrate": train_m["win_rate"],
            "train_trades": int(train_m["trades"]),
            "test_sharpe": test_m["sharpe"],
            "test_winrate": test_m["win_rate"],
            "test_trades": total,
            "test_wins": wins,
            "test_return": test_m["total_return"],
            "test_max_dd": test_m["max_dd"],
            "p_value": p_value,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, TuningResult(strat, 0, {}, {}, {}, {}, 0.0, 0, 0)

    df["sig_bh_10"] = benjamini_hochberg(df["p_value"].to_numpy(), fdr=0.10)
    df["sig_bh_05"] = benjamini_hochberg(df["p_value"].to_numpy(), fdr=0.05)

    # Train-set best (HONEST: pick by train, report on test)
    train_best_idx = df["train_sharpe"].idxmax()
    train_best_row = df.loc[train_best_idx]
    train_best_params = {k: train_best_row[k] for k in grid.keys()}

    # Cherry-pick (test-set best — for diagnostic only, has selection bias)
    test_best_idx = df["test_sharpe"].idxmax()
    test_best_row = df.loc[test_best_idx]

    deflation = float((df["train_sharpe"] - df["test_sharpe"]).mean())

    summary = TuningResult(
        strategy=strat,
        grid_size=len(df),
        train_best_params=train_best_params,
        test_metrics_at_train_best={
            "test_sharpe": float(train_best_row["test_sharpe"]),
            "test_winrate": float(train_best_row["test_winrate"]),
            "test_trades": int(train_best_row["test_trades"]),
            "test_return": float(train_best_row["test_return"]),
            "p_value": float(train_best_row["p_value"]),
        },
        cherry_pick_test_best={
            "test_sharpe": float(test_best_row["test_sharpe"]),
            "test_winrate": float(test_best_row["test_winrate"]),
            "test_trades": int(test_best_row["test_trades"]),
            "test_return": float(test_best_row["test_return"]),
            "p_value": float(test_best_row["p_value"]),
        },
        cherry_pick_test_best_params={k: test_best_row[k] for k in grid.keys()},
        train_test_sharpe_deflation=deflation,
        n_significant_bh10=int(df["sig_bh_10"].sum()),
        n_significant_bh05=int(df["sig_bh_05"].sum()),
    )
    return df, summary


# ─── Reporting ─────────────────────────────────────────────────────────
def fmt_params(d: dict) -> str:
    return ", ".join(f"{k}={int(v) if isinstance(v, (int, np.integer)) else v}" for k, v in d.items())


def print_summary(symbol: str, all_summaries: list[TuningResult], all_results: dict[str, pd.DataFrame]):
    total_combos = sum(s.grid_size for s in all_summaries)
    total_sig_10 = sum(s.n_significant_bh10 for s in all_summaries)
    total_sig_05 = sum(s.n_significant_bh05 for s in all_summaries)

    print(f"\n{'='*78}")
    print(f"  TUNING SUMMARY: {symbol}")
    print(f"{'='*78}\n")
    print(f"  Total grid points tested: {total_combos}")
    print(f"  Significant at BH FDR=10%: {total_sig_10} / {total_combos} ({total_sig_10/total_combos*100:.1f}%)")
    print(f"  Significant at BH FDR= 5%: {total_sig_05} / {total_combos} ({total_sig_05/total_combos*100:.1f}%)")

    print(f"\n  {'─'*76}")
    print(f"  HONEST: train-set-best params, evaluated on held-out test")
    print(f"  {'─'*76}")
    print(f"  {'strategy':<18} {'test_sharpe':>11} {'test_win':>9} {'test_ret':>9} {'p-value':>9} {'BH 10%':>7}")
    for s in all_summaries:
        m = s.test_metrics_at_train_best
        sig = "yes" if m["p_value"] <= s.n_significant_bh10 / s.grid_size * 0.10 else "no"
        print(
            f"  {s.strategy:<18} {m['test_sharpe']:>+10.2f}  {m['test_winrate']*100:>7.1f}%  "
            f"{m['test_return']*100:>+7.1f}%  {m['p_value']:>9.4f}  {sig:>7}"
        )
        print(f"     params: {fmt_params(s.train_best_params)}")

    print(f"\n  {'─'*76}")
    print(f"  CHERRY-PICKED: best-on-test params (HAS SELECTION BIAS — for comparison only)")
    print(f"  {'─'*76}")
    print(f"  {'strategy':<18} {'test_sharpe':>11} {'test_win':>9} {'test_ret':>9} {'naive p':>9}")
    for s in all_summaries:
        c = s.cherry_pick_test_best
        print(
            f"  {s.strategy:<18} {c['test_sharpe']:>+10.2f}  {c['test_winrate']*100:>7.1f}%  "
            f"{c['test_return']*100:>+7.1f}%  {c['p_value']:>9.4f}"
        )
        print(f"     params: {fmt_params(s.cherry_pick_test_best_params)}")

    print(f"\n  {'─'*76}")
    print(f"  OVERFITTING: train_sharpe minus test_sharpe (per strategy, averaged across grid)")
    print(f"  {'─'*76}")
    print(f"  {'strategy':<18} {'mean deflation':>16}  interpretation")
    for s in all_summaries:
        d = s.train_test_sharpe_deflation
        if d > 0.5:
            note = "severe overfitting"
        elif d > 0.2:
            note = "moderate overfitting"
        elif d > 0.0:
            note = "mild overfitting"
        else:
            note = "no overfitting (test >= train)"
        print(f"  {s.strategy:<18} {d:>+15.3f}    {note}")

    # All BH-significant combos
    sig_combos = []
    for strat, df in all_results.items():
        if df.empty:
            continue
        for _, row in df[df["sig_bh_10"]].iterrows():
            sig_combos.append({
                "strategy": strat,
                **{k: row[k] for k in GRIDS[strat].keys()},
                "test_sharpe": row["test_sharpe"],
                "test_winrate": row["test_winrate"],
                "test_trades": int(row["test_trades"]),
                "p_value": row["p_value"],
            })

    if sig_combos:
        sig_df = pd.DataFrame(sig_combos).sort_values("test_sharpe", ascending=False)
        print(f"\n  {'─'*76}")
        print(f"  ROBUSTLY SIGNIFICANT COMBOS (survived BH FDR=10% across {total_combos} tests)")
        print(f"  {'─'*76}")
        for _, row in sig_df.head(15).iterrows():
            params = {k: row[k] for k in GRIDS[row["strategy"]].keys()}
            print(
                f"  {row['strategy']:<18} {fmt_params(params):<48} "
                f"sh={row['test_sharpe']:+.2f}  win={row['test_winrate']*100:.1f}%  "
                f"n={int(row['test_trades']):>3d}  p={row['p_value']:.4f}"
            )
    else:
        print(f"\n  No combos survived BH correction at FDR=10%. The grid contains only noise.")

    print(f"\n{'='*78}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--train_frac", type=float, default=0.6)
    args = parser.parse_args()

    start = (date.today() - timedelta(days=args.years * 365)).isoformat()
    df = sources.get_bars(args.symbol, start, use_cache=True)
    df = indicators.attach_all(df)

    print(f"\nSymbol: {args.symbol}  ·  bars: {len(df)}  ·  "
          f"period: {df.index.min().date()} -> {df.index.max().date()}  ·  "
          f"train/test split: {int(args.train_frac*100)}%/{int((1-args.train_frac)*100)}%")

    all_results = {}
    all_summaries = []

    for strat, grid in GRIDS.items():
        n = sum(1 for p in grid_combos(grid) if is_valid(strat, p))
        print(f"  Testing {strat}: {n} valid parameter combinations...")
        results, summary = tune_strategy(df, strat, grid, train_frac=args.train_frac)
        all_results[strat] = results
        all_summaries.append(summary)

    print_summary(args.symbol, all_summaries, all_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
