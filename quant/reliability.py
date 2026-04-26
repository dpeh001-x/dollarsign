"""Reliability assessment of the strategies + ensemble.

Four tests, each answers a different reliability question:

  1. Cross-sectional   — does it work across many symbols, or just SPY?
  2. Walk-forward      — does it survive proper out-of-sample testing where the
                         regime classifier is refit using only past data?
  3. Bootstrap CI      — what's the 95% confidence interval on win rate?
                         (Tells you if results are robust or sample-dependent)
  4. Random baseline   — does it actually beat random position-shuffling?
                         (Z-score test against the null of "no signal")

Run with:
    python reliability.py
"""
from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import sources
import indicators
import regime
import backtest
import strategies

logging.basicConfig(level=logging.WARNING)
np.random.seed(42)

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "SPY", "QQQ", "IWM", "DIA",
    "XLF", "XLE", "XLK", "XLV", "XLY", "XLP", "XLI",
    "EFA", "EEM", "TLT", "GLD", "USO",
    "BTC-USD", "ETH-USD",
]

STRATEGY_FNS = {
    "regime_breakout": strategies.regime_aware_breakout,
    "rsi_meanrev": strategies.rsi_mean_reversion,
    "ma_cross": strategies.ma_crossover,
    "xgb_wf": strategies.xgb_walk_forward,
}

STRATS = list(STRATEGY_FNS) + ["ensemble"]


def positions_for(strat: str, df: pd.DataFrame) -> pd.Series:
    if strat == "ensemble":
        # Ensemble = XGB only (rule-based strategies have demonstrably no edge,
        # so including their votes drags the ensemble down to noise. The rules
        # remain in STRATS for individual reporting / sanity-check transparency.)
        return STRATEGY_FNS["xgb_wf"](df)
    return STRATEGY_FNS[strat](df)


def prepare(symbol: str, start: str, regime_model: regime.RegimeModel | None = None):
    df = sources.get_bars(symbol, start, use_cache=True)
    df = indicators.attach_all(df)
    if regime_model is None:
        regime_model = regime.fit(df)
    df["regime"] = regime_model.predict(df)
    return df, regime_model


def run_strategy(df: pd.DataFrame, strat: str) -> backtest.BacktestResult:
    return backtest.run(df, positions_for(strat, df), fee_bps=1.0, slippage_bps=1.0)


# ─── 1. CROSS-SECTIONAL ────────────────────────────────────────────────
def cross_sectional() -> tuple[pd.DataFrame, dict[str, list[pd.DataFrame]]]:
    """Run each strategy on each symbol. Return (metrics_df, trades_pool_by_strategy)."""
    rows: list[dict] = []
    trade_pool: dict[str, list[pd.DataFrame]] = {s: [] for s in STRATS}
    start = (date.today() - timedelta(days=4 * 365)).isoformat()

    for sym in UNIVERSE:
        try:
            df, _ = prepare(sym, start)
        except Exception as e:
            print(f"  skip {sym}: {type(e).__name__}: {str(e)[:60]}")
            continue
        if df.empty or len(df) < 100:
            print(f"  skip {sym}: insufficient bars ({len(df)})")
            continue

        for strat in STRATS:
            try:
                r = run_strategy(df, strat)
                m = r.metrics
                rows.append({
                    "symbol": sym,
                    "strategy": strat,
                    "trades": int(m["trades"]),
                    "win_rate": m["win_rate"],
                    "total_return": m["total_return"],
                    "sharpe": m["sharpe"],
                    "max_dd": m["max_dd"],
                    "n_bars": int(m["bars"]),
                })
                if len(r.trades) > 0:
                    trade_pool[strat].append(r.trades.assign(symbol=sym))
            except Exception as e:
                print(f"  {sym} {strat} failed: {e}")

    return pd.DataFrame(rows), trade_pool


# ─── 2. WALK-FORWARD ───────────────────────────────────────────────────
def walk_forward(symbol: str, train_years: int = 2, test_years: int = 1) -> pd.DataFrame:
    """Refit regime classifier per window using only prior data; test forward."""
    full_start = (date.today() - timedelta(days=8 * 365)).isoformat()
    df_full = sources.get_bars(symbol, full_start, use_cache=True)
    df_full = indicators.attach_all(df_full)

    train_bars = train_years * 252
    test_bars = test_years * 252
    warmup = 60  # for indicator state

    rows: list[dict] = []
    i = train_bars
    while i + test_bars < len(df_full):
        train_df = df_full.iloc[:i]
        test_window = df_full.iloc[i - warmup: i + test_bars].copy()

        try:
            model = regime.fit(train_df)
        except Exception:
            i += test_bars
            continue

        test_window["regime"] = model.predict(test_window)
        test_period = test_window.iloc[warmup:]  # trim warmup

        for strat in STRATS:
            try:
                r = run_strategy(test_period, strat)
                m = r.metrics
                rows.append({
                    "period_end": test_period.index[-1].date(),
                    "strategy": strat,
                    "trades": int(m["trades"]),
                    "win_rate": m["win_rate"],
                    "total_return": m["total_return"],
                    "sharpe": m["sharpe"],
                })
            except Exception:
                pass

        i += test_bars
    return pd.DataFrame(rows)


# ─── 3. BOOTSTRAP CI ────────────────────────────────────────────────────
def bootstrap_win_rate_ci(trades: pd.DataFrame, n_iter: int = 2000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Bootstrap (1-alpha) confidence interval on win rate."""
    if len(trades) == 0:
        return (np.nan, np.nan, np.nan)
    wins = (trades["net_pct"] > 0).astype(float).to_numpy()
    n = len(wins)
    rng = np.random.default_rng(42)
    boot = np.array([wins[rng.integers(0, n, n)].mean() for _ in range(n_iter)])
    return (
        float(boot.mean()),
        float(np.percentile(boot, alpha / 2 * 100)),
        float(np.percentile(boot, (1 - alpha / 2) * 100)),
    )


# ─── 4. RANDOM BASELINE ─────────────────────────────────────────────────
def random_baseline(symbol: str = "SPY", n_iter: int = 200) -> dict:
    """Z-test: actual ensemble vs random-shuffled positions (preserves marginals)."""
    df, _ = prepare(symbol, (date.today() - timedelta(days=3 * 365)).isoformat())
    real_pos = positions_for("ensemble", df)
    real_wr = backtest.run(df, real_pos, fee_bps=1.0, slippage_bps=1.0).metrics["win_rate"]

    rng = np.random.default_rng(42)
    pos_arr = real_pos.to_numpy().copy()

    random_wrs: list[float] = []
    for _ in range(n_iter):
        rng.shuffle(pos_arr)
        shuffled = pd.Series(pos_arr, index=real_pos.index)
        r = backtest.run(df, shuffled, fee_bps=1.0, slippage_bps=1.0)
        if r.metrics["trades"] > 0:
            random_wrs.append(r.metrics["win_rate"])

    if not random_wrs:
        return {"real_win": real_wr, "random_mean": np.nan, "random_std": np.nan, "z": np.nan, "n": 0}

    rand_mean = float(np.mean(random_wrs))
    rand_std = float(np.std(random_wrs))
    z = (real_wr - rand_mean) / rand_std if rand_std > 0 else float("nan")
    return {
        "real_win": real_wr,
        "random_mean": rand_mean,
        "random_std": rand_std,
        "z": z,
        "n": len(random_wrs),
    }


# ─── REPORT ─────────────────────────────────────────────────────────────
def print_section_1(cs: pd.DataFrame) -> None:
    summary = cs.groupby("strategy").agg(
        win_med=("win_rate", "median"),
        win_p25=("win_rate", lambda x: x.quantile(0.25)),
        win_p75=("win_rate", lambda x: x.quantile(0.75)),
        trades_med=("trades", "median"),
        ret_med=("total_return", "median"),
        sharpe_med=("sharpe", "median"),
    )
    pct_profit = cs.groupby("strategy")["total_return"].apply(lambda x: (x > 0).mean())
    pct_win = cs.groupby("strategy")["win_rate"].apply(lambda x: (x >= 0.5).mean())

    print(f"\n  Strategy win-rate distribution across {cs['symbol'].nunique()} symbols:")
    print(f"  {'strategy':<18} {'win_med':>8} {'win_IQR':>14} {'tr_med':>7} {'ret_med':>9} {'sh_med':>7} {'%>50%win':>9} {'%profit':>9}")
    print(f"  {'-'*18} {'-'*8} {'-'*14} {'-'*7} {'-'*9} {'-'*7} {'-'*9} {'-'*9}")
    for strat in STRATS:
        if strat not in summary.index:
            continue
        s = summary.loc[strat]
        iqr = f"{s.win_p25:.1%}-{s.win_p75:.1%}"
        print(
            f"  {strat:<18} {s.win_med:>7.1%} {iqr:>14} {int(s.trades_med):>7d} "
            f"{s.ret_med:>+8.1%} {s.sharpe_med:>7.2f} {pct_win[strat]:>8.0%} {pct_profit[strat]:>8.0%}"
        )


def print_section_2(wf: pd.DataFrame) -> None:
    if len(wf) == 0:
        print("  (no walk-forward periods completed)")
        return
    summary = wf.groupby("strategy").agg(
        win_mean=("win_rate", "mean"),
        win_std=("win_rate", "std"),
        ret_mean=("total_return", "mean"),
        sharpe_mean=("sharpe", "mean"),
        trades_total=("trades", "sum"),
    )
    print(f"\n  Walk-forward stability ({wf['period_end'].nunique()} held-out periods):")
    print(f"  {'strategy':<18} {'win_mean':>9} {'win_std':>9} {'ret_mean':>9} {'sharpe_mean':>11} {'trades':>8}")
    print(f"  {'-'*18} {'-'*9} {'-'*9} {'-'*9} {'-'*11} {'-'*8}")
    for strat in STRATS:
        if strat not in summary.index:
            continue
        s = summary.loc[strat]
        print(
            f"  {strat:<18} {s.win_mean:>8.1%} {s.win_std:>8.2%} {s.ret_mean:>+8.1%} "
            f"{s.sharpe_mean:>10.2f} {int(s.trades_total):>8d}"
        )


def print_section_3(trade_pool: dict[str, list[pd.DataFrame]]) -> None:
    print(f"\n  {'strategy':<18} {'pooled_trades':>14} {'win_mean':>9} {'95% CI':>20}")
    print(f"  {'-'*18} {'-'*14} {'-'*9} {'-'*20}")
    for strat in STRATS:
        trade_list = trade_pool.get(strat, [])
        if not trade_list:
            print(f"  {strat:<18} {'0':>14}  no data")
            continue
        all_trades = pd.concat(trade_list, ignore_index=True)
        mean_w, lo, hi = bootstrap_win_rate_ci(all_trades, n_iter=2000)
        print(f"  {strat:<18} {len(all_trades):>14d} {mean_w:>8.1%}   [{lo:.1%}, {hi:.1%}]")


def print_section_4(rb: dict) -> None:
    if not rb or rb.get("n", 0) == 0:
        print("  (random baseline failed)")
        return
    print(f"\n  Real ensemble win rate:    {rb['real_win']:.1%}")
    print(f"  Random shuffle baseline:   {rb['random_mean']:.1%}  (sd {rb['random_std']:.2%}, n={rb['n']})")
    z = rb["z"]
    print(f"  Z-score:                   {z:+.2f}")
    if z > 1.96:
        verdict = "STATISTICALLY BETTER than random shuffling (p < 0.05)"
    elif z < -1.96:
        verdict = "STATISTICALLY WORSE than random (p < 0.05)"
    else:
        verdict = "indistinguishable from random shuffling"
    print(f"  Verdict:                   {verdict}")


def main() -> int:
    print("\n" + "=" * 72)
    print("  RELIABILITY ASSESSMENT")
    print("=" * 72)

    print(f"\n[1/4] Cross-sectional: {len(STRATS)} strategies x {len(UNIVERSE)} symbols x ~4y daily...")
    cs, trade_pool = cross_sectional()
    print(f"      Completed: {len(cs)} (symbol, strategy) cells, "
          f"{cs['symbol'].nunique()} symbols loaded")
    print_section_1(cs)

    print(f"\n[2/4] Walk-forward on SPY (2y train -> 1y test, sliding)...")
    wf = walk_forward("SPY", train_years=2, test_years=1)
    print_section_2(wf)

    print(f"\n[3/4] Bootstrap 95% CI on pooled-universe win rate (2000 resamples)...")
    print_section_3(trade_pool)

    print(f"\n[4/4] Random-shuffle baseline on SPY (200 shuffles)...")
    rb = random_baseline("SPY", n_iter=200)
    print_section_4(rb)

    print("\n" + "=" * 72)
    print("  HOW TO READ THIS")
    print("=" * 72)
    print("""
  win_med / win_IQR    Median + interquartile range across the 25-symbol universe.
                       A reliable strategy has high median AND tight IQR (consistent).
  %>50% win            Fraction of symbols where strategy wins more often than not.
  %profit              Fraction of symbols where strategy ends with positive return.
  walk-forward         Out-of-sample: regime classifier sees only past data.
                       win_std measures stability — high std = unreliable across
                       different market periods.
  95% CI on win rate   Bootstrap interval. If lower bound > 50%, strategy reliably
                       wins more than half its trades. If CI brackets 50%, marginal.
  z-score vs random    > 1.96 = statistically distinguishable from random (p < .05).
                       <= 1.96 = strategy could be no better than coin-flipping.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
