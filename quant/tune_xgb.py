"""XGBoost hyperparameter tuning with Optuna Bayesian search + selection-bias controls.

Replaces the original exhaustive grid search with Optuna's Tree-structured
Parzen Estimator (TPE). This explores a much larger continuous search space
(~50 trials vs 36 fixed grid combos) and tunes the long_thresh / short_thresh
deadband directly — previously hardcoded at ±0.075 from 0.5.

Same statistical hygiene as before:
  - Time-respecting train/test split
  - Train-set selection then honest test eval
  - Benjamini-Hochberg FDR correction across all trials

Run:
    python tune_xgb.py --symbol SPY --years 8 --n_trials 50
"""
from __future__ import annotations

import argparse
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
    long_thresh = 0.5 + params["deadband"]
    short_thresh = 0.5 - params["deadband"]
    xgb_params = {
        "max_depth": params["max_depth"],
        "n_estimators": params["n_estimators"],
        "learning_rate": params["learning_rate"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
    }

    model = ml.WalkForwardModel(
        train_size=train_size,
        step_size=step_size,
        horizon=params["horizon"],
        long_thresh=long_thresh,
        short_thresh=short_thresh,
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
    n_trials: int = 50,
) -> pd.DataFrame:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Optuna required: pip install optuna>=3.6.0")

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

    print(f"  train period: {len(train_period)} bars  ({train_period.index.min().date()} -> {train_period.index.max().date()})")
    print(f"  test period:  {len(test_period)} bars  ({test_period.index.min().date()} -> {test_period.index.max().date()})")
    print(f"  running {n_trials} Optuna TPE trials\n")

    rows: list[dict] = []
    t0 = time.time()

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "horizon":        trial.suggest_categorical("horizon", [3, 5, 10]),
            "deadband":       trial.suggest_float("deadband", 0.03, 0.15),
            "max_depth":      trial.suggest_int("max_depth", 2, 6),
            "n_estimators":   trial.suggest_int("n_estimators", 50, 300, step=50),
            "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":      trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        try:
            train_m, test_m, _, test_trades = evaluate_combo(
                df, params, cut_idx, train_size=train_size, step_size=step_size
            )
        except Exception:
            return float("-inf")

        if test_m["trades"] > 0 and len(test_trades) > 0:
            wins = int((test_trades["net_pct"] > 0).sum())
            total = int(test_m["trades"])
            p_value = winrate_pvalue(wins, total)
        else:
            wins, total, p_value = 0, 0, 1.0

        elapsed = time.time() - t0
        trial_n = trial.number + 1
        eta = elapsed / trial_n * (n_trials - trial_n)
        print(
            f"  [{trial_n:>3}/{n_trials}] h={params['horizon']:>2d} db={params['deadband']:.3f} "
            f"d={params['max_depth']} n={params['n_estimators']:>3d} lr={params['learning_rate']:.3f} "
            f"-> train_sh={train_m['sharpe']:>+5.2f} test_sh={test_m['sharpe']:>+5.2f} "
            f"test_win={test_m['win_rate']*100:>5.1f}% n={int(test_m['trades']):>3d}  (eta {eta:.0f}s)"
        )

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

        # Objective: train-set Sharpe (honest selection — test Sharpe not seen during tuning)
        return float(train_m["sharpe"])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    df_results = pd.DataFrame(rows)
    if not df_results.empty:
        df_results["sig_bh_10"] = benjamini_hochberg(df_results["p_value"].to_numpy(), fdr=0.10)
        df_results["sig_bh_05"] = benjamini_hochberg(df_results["p_value"].to_numpy(), fdr=0.05)
    return df_results


def fmt_params(row: pd.Series) -> str:
    return (
        f"horizon={int(row['horizon'])}, deadband={row['deadband']:.3f}, "
        f"max_depth={int(row['max_depth'])}, n_estimators={int(row['n_estimators'])}, "
        f"learning_rate={row['learning_rate']:.4f}"
    )


def print_summary(symbol: str, df: pd.DataFrame) -> None:
    if df.empty:
        print("  (no results)")
        return

    n_combos = len(df)
    n_sig_10 = int(df["sig_bh_10"].sum())
    n_sig_05 = int(df["sig_bh_05"].sum())

    print(f"\n{'='*78}")
    print(f"  XGB TUNING SUMMARY (Optuna TPE): {symbol}")
    print(f"{'='*78}")
    print(f"\n  Total trials: {n_combos}")
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
    print(f"  Mean train_sharpe - test_sharpe across trials: {deflation:+.3f}")
    if deflation > 0.5:
        note = "SEVERE — model is fitting noise in training data"
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
        f"  {'sig':>3} {'horiz':>5} {'dband':>6} {'depth':>5} {'n_est':>5} {'lr':>6} | "
        f"{'tr_sh':>6} {'te_sh':>6} {'te_win':>7} {'te_ret':>8} {'p':>7}"
    )
    print(f"  {'-'*3} {'-'*5} {'-'*6} {'-'*5} {'-'*5} {'-'*6}-+-{'-'*6}-{'-'*6}-{'-'*7}-{'-'*8}-{'-'*7}")
    for _, r in top.iterrows():
        sig = " * " if r["sig_bh_10"] else "   "
        print(
            f"  {sig} {int(r['horizon']):>5d} {r['deadband']:>6.3f} {int(r['max_depth']):>5d} "
            f"{int(r['n_estimators']):>5d} {r['learning_rate']:>6.4f} | "
            f"{r['train_sharpe']:>+5.2f} {r['test_sharpe']:>+5.2f} "
            f"{r['test_winrate']*100:>6.1f}% {r['test_return']*100:>+7.1f}% {r['p_value']:>7.4f}"
        )

    # All BH-significant
    sig_combos = df[df["sig_bh_10"]].sort_values("test_sharpe", ascending=False)
    if len(sig_combos) > 0:
        print(f"\n  ─── ROBUSTLY SIGNIFICANT TRIALS (survived BH FDR=10%) ───")
        for _, r in sig_combos.iterrows():
            print(
                f"  {fmt_params(r)}\n"
                f"     -> sharpe={r['test_sharpe']:+.2f}  win={r['test_winrate']*100:.1f}%  "
                f"trades={int(r['test_trades'])}  ret={r['test_return']*100:+.1f}%  p={r['p_value']:.4f}"
            )
    else:
        print(f"\n  No trials survived BH correction at FDR=10%.")
        print(f"  Either the search found no real edge, or sample size is too small to detect it.")

    print(f"\n{'='*78}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--train_frac", type=float, default=0.6)
    parser.add_argument("--train_size", type=int, default=500, help="WF train_size")
    parser.add_argument("--step_size", type=int, default=60, help="WF step_size")
    parser.add_argument("--n_trials", type=int, default=50, help="Optuna trial count")
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
        n_trials=args.n_trials,
    )
    print_summary(args.symbol, df_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
