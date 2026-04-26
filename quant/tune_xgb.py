"""XGBoost hyperparameter grid-search with selection-bias controls.

Same statistical hygiene as tuning.py — time-respecting train/test split,
train-set selection then honest test eval, Benjamini-Hochberg FDR
correction across the grid.

Key implementation choice: we run WalkForwardModel ONCE on the full
history per hyperparameter combo, then split its predictions into
train and test portions for evaluation. This is more realistic than
running WF separately on each portion (matches how you'd use it in
production: train on all available history, predict forward).

The regime classifier is fit on the train portion only, then applied
to the full df — no future-leak into the regime feature.

Tunable hyperparameters:
    horizon      — N-day forward return horizon (target definition)
    deadband     — long_thresh = 0.5 + deadband, short_thresh = 0.5 - deadband
    max_depth    — XGB tree depth
    n_estimators — XGB number of trees
    learning_rate — XGB learning rate

The internal WF train_size and step_size are not tuned here (fixed at
500 / 60). Add them to GRID and rerun if you want to explore.

Run:
    python tune_xgb.py --symbol SPY --years 8
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import sources
import indicators
import regime
import backtest
import ml

logging.basicConfig(level=logging.WARNING)


XGB_GRID: dict[str, list] = {
    "horizon": [3, 5, 10],
    "deadband": [0.05, 0.08, 0.10],
    "max_depth": [3, 5],
    "n_estimators": [100, 200],
    "learning_rate": [0.05],
}


def grid_combos(grid):
    keys = list(grid.keys())
    for combo in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, combo))


def winrate_pvalue(wins: int, total: int) -> float:
    if total == 0:
        return 1.0
    return stats.binomtest(wins, total, p=0.5, alternative="two-sided").pvalue


def benjamini_hochberg(pvalues: np.ndarray, fdr: float = 0.10) -> np.ndarray:
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


def evaluate_combo(
    df: pd.DataFrame,
    params: dict,
    cut_idx: int,
    train_size: int = 500,
    step_size: int = 60,
) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    """Run WF on full df, split predictions into train/test portions, backtest each."""
    deadband = params["deadband"]
    xgb_params = {
        "max_depth": params["max_depth"],
        "n_estimators": params["n_estimators"],
        "learning_rate": params["learning_rate"],
    }

    model = ml.WalkForwardModel(
        train_size=train_size,
        step_size=step_size,
        horizon=params["horizon"],
        long_thresh=0.5 + deadband,
        short_thresh=0.5 - deadband,
        xgb_params=xgb_params,
    )
    positions = model.positions(df)

    train_df = df.iloc[:cut_idx]
    test_df = df.iloc[cut_idx:]
    train_pos = positions.iloc[:cut_idx]
    test_pos = positions.iloc[cut_idx:]

    train_r = backtest.run(train_df, train_pos, fee_bps=1.0, slippage_bps=1.0)
    test_r = backtest.run(test_df, test_pos, fee_bps=1.0, slippage_bps=1.0)

    return train_r.metrics, test_r.metrics, train_r.trades, test_r.trades


def tune_xgb(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    train_size: int = 500,
    step_size: int = 60,
) -> pd.DataFrame:
    cut_idx = int(len(df) * train_frac)
    train_period = df.iloc[:cut_idx]
    test_period = df.iloc[cut_idx:]

    if cut_idx < train_size + 50:
        raise ValueError(
            f"Train period too short: {cut_idx} bars vs WF train_size={train_size}. "
            "Increase --years or reduce --train_size."
        )
    if len(test_period) < 100:
        raise ValueError(f"Test period too short: {len(test_period)} bars. Increase --years.")

    # Fit regime model on TRAIN only; apply to full df → no future-leak
    regime_model = regime.fit(train_period)
    df = df.copy()
    df["regime"] = regime_model.predict(df)

    n_combos = sum(1 for _ in grid_combos(XGB_GRID))
    print(f"  train period: {len(train_period)} bars  ({train_period.index.min().date()} -> {train_period.index.max().date()})")
    print(f"  test period:  {len(test_period)} bars  ({test_period.index.min().date()} -> {test_period.index.max().date()})")
    print(f"  testing {n_combos} XGB hyperparameter combinations\n")

    rows: list[dict] = []
    t0 = time.time()
    for i, params in enumerate(grid_combos(XGB_GRID)):
        try:
            train_m, test_m, _, test_trades = evaluate_combo(
                df, params, cut_idx, train_size=train_size, step_size=step_size
            )
        except Exception as e:
            print(f"  [{i+1:>2}/{n_combos}] {params}: failed ({e})")
            continue

        if test_m["trades"] > 0 and len(test_trades) > 0:
            wins = int((test_trades["net_pct"] > 0).sum())
            total = int(test_m["trades"])
            p_value = winrate_pvalue(wins, total)
        else:
            wins, total, p_value = 0, 0, 1.0

        rows.append({
            **params,
            "train_sharpe": train_m["sharpe"],
            "train_winrate": train_m["win_rate"],
            "train_trades": int(train_m["trades"]),
            "train_return": train_m["total_return"],
            "test_sharpe": test_m["sharpe"],
            "test_winrate": test_m["win_rate"],
            "test_trades": total,
            "test_wins": wins,
            "test_return": test_m["total_return"],
            "test_max_dd": test_m["max_dd"],
            "p_value": p_value,
        })

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (n_combos - i - 1)
        print(
            f"  [{i+1:>2}/{n_combos}] h={params['horizon']:>2d} db={params['deadband']:>4.2f} "
            f"d={params['max_depth']} n={params['n_estimators']:>3d} lr={params['learning_rate']:.2f} "
            f"-> train_sh={train_m['sharpe']:>+5.2f} test_sh={test_m['sharpe']:>+5.2f} "
            f"test_win={test_m['win_rate']*100:>5.1f}% n={int(test_m['trades']):>3d}  (eta {eta:.0f}s)"
        )

    df_results = pd.DataFrame(rows)
    if not df_results.empty:
        df_results["sig_bh_10"] = benjamini_hochberg(df_results["p_value"].to_numpy(), fdr=0.10)
        df_results["sig_bh_05"] = benjamini_hochberg(df_results["p_value"].to_numpy(), fdr=0.05)
    return df_results


def fmt_params(row: pd.Series) -> str:
    return (
        f"horizon={int(row['horizon'])}, deadband={row['deadband']:.2f}, "
        f"max_depth={int(row['max_depth'])}, n_estimators={int(row['n_estimators'])}, "
        f"learning_rate={row['learning_rate']:.2f}"
    )


def print_summary(symbol: str, df: pd.DataFrame) -> None:
    if df.empty:
        print("  (no results)")
        return

    n_combos = len(df)
    n_sig_10 = int(df["sig_bh_10"].sum())
    n_sig_05 = int(df["sig_bh_05"].sum())

    print(f"\n{'='*78}")
    print(f"  XGB TUNING SUMMARY: {symbol}")
    print(f"{'='*78}")
    print(f"\n  Total grid points: {n_combos}")
    print(f"  Significant at BH FDR=10%: {n_sig_10} ({n_sig_10/n_combos*100:.1f}%)")
    print(f"  Significant at BH FDR= 5%: {n_sig_05} ({n_sig_05/n_combos*100:.1f}%)")

    # Honest train-set best
    train_best_idx = df["train_sharpe"].idxmax()
    tb = df.loc[train_best_idx]
    print(f"\n  ─── HONEST: train-set-best (selected on train, evaluated on test) ───")
    print(f"  params: {fmt_params(tb)}")
    print(
        f"  train: sharpe={tb['train_sharpe']:+.2f}  win={tb['train_winrate']*100:.1f}%  "
        f"trades={int(tb['train_trades'])}  ret={tb['train_return']*100:+.1f}%"
    )
    print(
        f"  test:  sharpe={tb['test_sharpe']:+.2f}  win={tb['test_winrate']*100:.1f}%  "
        f"trades={int(tb['test_trades'])}  ret={tb['test_return']*100:+.1f}%  "
        f"p={tb['p_value']:.4f}  BH 10% sig: {bool(tb['sig_bh_10'])}"
    )

    # Cherry-picked test best (selection bias)
    test_best_idx = df["test_sharpe"].idxmax()
    cb = df.loc[test_best_idx]
    print(f"\n  ─── CHERRY-PICKED: best-on-test (HAS SELECTION BIAS) ───")
    print(f"  params: {fmt_params(cb)}")
    print(
        f"  test:  sharpe={cb['test_sharpe']:+.2f}  win={cb['test_winrate']*100:.1f}%  "
        f"trades={int(cb['test_trades'])}  ret={cb['test_return']*100:+.1f}%  "
        f"p={cb['p_value']:.4f}  BH 10% sig: {bool(cb['sig_bh_10'])}"
    )

    # Overfitting deflation
    deflation = float((df["train_sharpe"] - df["test_sharpe"]).mean())
    print(f"\n  ─── OVERFITTING ───")
    print(f"  Mean train_sharpe - test_sharpe across grid: {deflation:+.3f}")
    if deflation > 0.5:
        note = "SEVERE — XGB is fitting noise in training data"
    elif deflation > 0.2:
        note = "moderate"
    elif deflation > 0:
        note = "mild (expected)"
    else:
        note = "none — test performance >= train (unusual; could be regime shift)"
    print(f"  Verdict: {note}")

    # Top by test sharpe
    print(f"\n  ─── TOP 10 BY TEST SHARPE (* = BH-significant at FDR 10%) ───")
    top = df.sort_values("test_sharpe", ascending=False).head(10)
    print(
        f"  {'sig':>3} {'horiz':>5} {'dband':>5} {'depth':>5} {'n_est':>5} {'lr':>5} | "
        f"{'tr_sh':>6} {'te_sh':>6} {'te_win':>7} {'te_ret':>8} {'p':>7}"
    )
    print(f"  {'-'*3} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5}-+-{'-'*6}-{'-'*6}-{'-'*7}-{'-'*8}-{'-'*7}")
    for _, r in top.iterrows():
        sig = " * " if r["sig_bh_10"] else "   "
        print(
            f"  {sig} {int(r['horizon']):>5d} {r['deadband']:>5.2f} {int(r['max_depth']):>5d} "
            f"{int(r['n_estimators']):>5d} {r['learning_rate']:>5.2f} | "
            f"{r['train_sharpe']:>+5.2f} {r['test_sharpe']:>+5.2f} "
            f"{r['test_winrate']*100:>6.1f}% {r['test_return']*100:>+7.1f}% {r['p_value']:>7.4f}"
        )

    # All BH-significant
    sig_combos = df[df["sig_bh_10"]].sort_values("test_sharpe", ascending=False)
    if len(sig_combos) > 0:
        print(f"\n  ─── ROBUSTLY SIGNIFICANT COMBOS (survived BH FDR=10%) ───")
        for _, r in sig_combos.iterrows():
            print(
                f"  {fmt_params(r)}\n"
                f"     -> sharpe={r['test_sharpe']:+.2f}  win={r['test_winrate']*100:.1f}%  "
                f"trades={int(r['test_trades'])}  ret={r['test_return']*100:+.1f}%  p={r['p_value']:.4f}"
            )
    else:
        print(f"\n  No combos survived BH correction at FDR=10%.")
        print(f"  Either the grid contains no real edge, or sample size is too small to detect it.")

    print(f"\n{'='*78}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--train_frac", type=float, default=0.6)
    parser.add_argument("--train_size", type=int, default=500, help="WF train_size")
    parser.add_argument("--step_size", type=int, default=60, help="WF step_size")
    args = parser.parse_args()

    start = (date.today() - timedelta(days=args.years * 365)).isoformat()
    df = sources.get_bars(args.symbol, start, use_cache=True)
    df = indicators.attach_all(df)

    print(f"\nSymbol: {args.symbol}  ·  bars: {len(df)}  ·  "
          f"period: {df.index.min().date()} -> {df.index.max().date()}")

    df_results = tune_xgb(
        df,
        train_frac=args.train_frac,
        train_size=args.train_size,
        step_size=args.step_size,
    )
    print_summary(args.symbol, df_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
