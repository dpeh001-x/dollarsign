"""Hypothetical 6-month walk-forward portfolio: follow every signal with ATR stops.

For each symbol:
  1. Train the walk-forward XGB (or class-pooled XGB if symbol is in
     equities/broad_etf/crypto) using only data up to each prediction bar.
  2. Walk through the last 6 months bar-by-bar:
       - When flat & P(up)>0.55: enter LONG with stop at -2.5×ATR, TP at +2.5×ATR
       - When flat & P(up)<0.45: enter SHORT with stop at +2.5×ATR, TP at -2.5×ATR
       - When in position: check intraday hit on stop/TP, also exit at 10 trading
         days (time stop)
       - Costs: 1 bp fee + 1 bp slip per side
  3. Compound trade returns to get per-symbol equity curve.

Portfolio assumes equal weight (1/N per symbol). Saves a CSV of daily equity
to cache/hypothetical_6mo.csv for downstream charting.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import indicators
import regime
import ml
import class_xgb
from data import sources

# ─── Sim defaults (Setup B from 12-month validation) ───
# Validated config: 56.5% win, +16.9% return, Sharpe 0.97 over 12mo, 10 symbols.
# 30-bar time stop (vs 10) lets winners run — TP exits jumped 29% → 49%.
ATR_MULT = 2.5               # set by --atr-mult
TIME_STOP_BARS = 30          # set by --time-stop  (Setup B)
HORIZON = 10                 # set by --horizon (model's prediction horizon)
LONG_THRESH = 0.575          # set by --min-conf  (Setup B: 0.15 conf threshold)
SHORT_THRESH = 0.425         # set by --min-conf
FEE_BPS = 1.0
SLIP_BPS = 1.0
START_CAPITAL = 10_000.0
N_MONTHS = 6                 # set by --months

UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "BTC-USD", "ETH-USD", "XLE", "GLD"]


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    side: int       # +1 long, -1 short
    entry_price: float
    exit_price: float
    stop_price: float
    tp_price: float
    exit_reason: str
    bars_held: int
    pct_return: float  # net of 2-side costs


def simulate_one_symbol(symbol: str, df: pd.DataFrame, probas: pd.Series,
                        sim_start: pd.Timestamp) -> list[Trade]:
    """Walk through bars from sim_start onwards, executing the model's signals."""
    sim = df[df.index >= sim_start].copy()
    sim_proba = probas.reindex(sim.index)
    cost_per_side = (FEE_BPS + SLIP_BPS) / 10_000.0

    trades: list[Trade] = []
    in_pos = 0
    entry_price = stop_price = tp_price = 0.0
    entry_date = None
    bars_in_pos = 0

    for ts in sim.index:
        row = sim.loc[ts]
        proba = sim_proba.get(ts, np.nan)
        atr = row.get("atr_14", np.nan)
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])

        if pd.isna(atr) or close <= 0:
            continue

        # ─── Exit checks (when in position) ───
        if in_pos != 0:
            bars_in_pos += 1
            exit_price = None
            exit_reason = None

            if in_pos == 1:
                if low <= stop_price:
                    exit_price, exit_reason = stop_price, "stop"
                elif high >= tp_price:
                    exit_price, exit_reason = tp_price, "tp"
                elif bars_in_pos >= TIME_STOP_BARS:
                    exit_price, exit_reason = close, "time"
            else:  # short
                if high >= stop_price:
                    exit_price, exit_reason = stop_price, "stop"
                elif low <= tp_price:
                    exit_price, exit_reason = tp_price, "tp"
                elif bars_in_pos >= TIME_STOP_BARS:
                    exit_price, exit_reason = close, "time"

            if exit_price is not None:
                gross = (exit_price / entry_price - 1.0) * in_pos
                net = gross - 2 * cost_per_side
                trades.append(Trade(
                    symbol=symbol, entry_date=entry_date, exit_date=ts,
                    side=in_pos, entry_price=entry_price, exit_price=exit_price,
                    stop_price=stop_price, tp_price=tp_price,
                    exit_reason=exit_reason, bars_held=bars_in_pos, pct_return=net,
                ))
                in_pos = 0
                bars_in_pos = 0

        # ─── Entry on flat ───
        if in_pos == 0 and not pd.isna(proba):
            atr_pct = atr / close
            stop_dist = ATR_MULT * atr_pct
            if proba > LONG_THRESH:
                in_pos = 1
                entry_price, entry_date = close, ts
                stop_price = entry_price * (1 - stop_dist)
                tp_price = entry_price * (1 + stop_dist)
                bars_in_pos = 0
            elif proba < SHORT_THRESH:
                in_pos = -1
                entry_price, entry_date = close, ts
                stop_price = entry_price * (1 + stop_dist)
                tp_price = entry_price * (1 - stop_dist)
                bars_in_pos = 0

    # Close any open position at sim end
    if in_pos != 0 and entry_date is not None:
        last = sim.iloc[-1]
        gross = (float(last["close"]) / entry_price - 1.0) * in_pos
        net = gross - 2 * cost_per_side
        trades.append(Trade(
            symbol=symbol, entry_date=entry_date, exit_date=last.name,
            side=in_pos, entry_price=entry_price, exit_price=float(last["close"]),
            stop_price=stop_price, tp_price=tp_price,
            exit_reason="open_at_end", bars_held=bars_in_pos, pct_return=net,
        ))

    return trades


def equity_curve_from_trades(trades: list[Trade], date_index: pd.DatetimeIndex,
                             start: float = 1.0) -> pd.Series:
    """Step-function equity: compound trade returns at exit_date."""
    eq = pd.Series(start, index=date_index, dtype=float)
    cur = start
    for t in sorted(trades, key=lambda x: x.exit_date):
        cur *= (1 + t.pct_return)
        eq.loc[eq.index >= t.exit_date] = cur
    return eq


def get_probas_for_symbol(symbol: str, df: pd.DataFrame, full_start_iso: str) -> pd.Series:
    """Use class-pooled WF if symbol's class benefits, else per-symbol WF."""
    cls = class_xgb.classify_symbol(symbol)
    if cls in class_xgb.POOL_BENEFITS:
        probas_dict = class_xgb.predict_class(
            cls, full_start_iso, horizon=HORIZON, train_size=750, step_size=60
        )
        s = probas_dict.get(symbol, pd.Series(dtype=float))
        return s.reindex(df.index)
    else:
        wf = ml.WalkForwardModel(train_size=500, step_size=60, horizon=HORIZON)
        return wf.fit_predict(df)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Hypothetical portfolio sim with ATR stops + time stops.")
    parser.add_argument("--months", type=int, default=6, help="Months of history to simulate.")
    parser.add_argument("--min-conf", type=float, default=0.15,
                        help="Min model confidence to take a signal (Setup B default 0.15 = P>0.575/<0.425).")
    parser.add_argument("--horizon", type=int, default=10,
                        help="Model prediction horizon in trading days.")
    parser.add_argument("--time-stop", type=int, default=30,
                        help="Force-exit after this many bars in position (Setup B default 30).")
    parser.add_argument("--atr-mult", type=float, default=2.5,
                        help="Stop distance = atr_mult × ATR. TP set symmetrically (1:1 R:R).")
    args = parser.parse_args()

    global LONG_THRESH, SHORT_THRESH, N_MONTHS, HORIZON, TIME_STOP_BARS, ATR_MULT
    LONG_THRESH = 0.5 + args.min_conf / 2
    SHORT_THRESH = 0.5 - args.min_conf / 2
    N_MONTHS = args.months
    HORIZON = args.horizon
    TIME_STOP_BARS = args.time_stop
    ATR_MULT = args.atr_mult

    sim_start_dt = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=N_MONTHS)
    full_start_iso = (date.today() - timedelta(days=5 * 365)).isoformat()

    print(f"\n{'='*82}")
    print(f"  HYPOTHETICAL {N_MONTHS}-MONTH RUN")
    print(f"  Model horizon: {HORIZON}d  ·  Stop: {ATR_MULT}× ATR  ·  Time stop: {TIME_STOP_BARS} bars  ·  Costs: {FEE_BPS}+{SLIP_BPS} bps/side")
    print(f"  Signal thresholds: LONG > {LONG_THRESH:.3f} · SHORT < {SHORT_THRESH:.3f}  (min conf {args.min_conf:.0%})")
    print(f"  Sim window: {sim_start_dt.date()} → today · Universe: {len(UNIVERSE)} symbols")
    print(f"{'='*82}\n")

    per_sym: dict[str, dict] = {}
    all_trades: list[Trade] = []

    for sym in UNIVERSE:
        print(f"  {sym:<10s} ", end="", flush=True)
        try:
            df = sources.get_bars(sym, full_start_iso, use_cache=True)
            df = indicators.attach_all(df)
            df = regime.label(df)
        except Exception as e:
            print(f"prep failed: {e}")
            continue

        try:
            probas = get_probas_for_symbol(sym, df, full_start_iso)
        except Exception as e:
            print(f"predictions failed: {e}")
            continue

        if probas is None or probas.dropna().empty:
            print("no predictions")
            continue

        trades = simulate_one_symbol(sym, df, probas, sim_start_dt)
        sim_df = df[df.index >= sim_start_dt]
        eq = equity_curve_from_trades(trades, sim_df.index, start=1.0)
        per_sym[sym] = {"trades": trades, "equity": eq, "df_sim": sim_df}
        all_trades.extend(trades)

        n = len(trades)
        if n == 0:
            print("no trades fired")
            continue
        wins = sum(1 for t in trades if t.pct_return > 0)
        ret = (eq.iloc[-1] - 1) * 100
        bnh = (sim_df["close"].iloc[-1] / sim_df["close"].iloc[0] - 1) * 100
        reasons = {}
        for t in trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        rs = " ".join(f"{k[:3]}:{v}" for k, v in sorted(reasons.items()))
        print(f"{n:>3}t · win {wins/n*100:>4.0f}% · ret {ret:>+6.1f}% · b&h {bnh:>+6.1f}% · {rs}")

    if not per_sym:
        print("\nNo symbols traded. Aborting.")
        return 1

    # ─── Portfolio: equal-weight average of per-symbol equity ───
    print(f"\n{'='*82}")
    print(f"  PORTFOLIO (equal weight, 1/{len(per_sym)} per symbol)")
    print(f"{'='*82}\n")

    # Build a unified date index (union of all sim dates)
    all_dates = pd.DatetimeIndex(sorted(set().union(*(s["equity"].index for s in per_sym.values()))))
    portfolio = pd.DataFrame(index=all_dates)
    bnh_portfolio = pd.DataFrame(index=all_dates)
    for sym, s in per_sym.items():
        eq_aligned = s["equity"].reindex(all_dates).ffill().fillna(1.0)
        portfolio[sym] = eq_aligned
        bnh_eq = (s["df_sim"]["close"] / s["df_sim"]["close"].iloc[0]).reindex(all_dates).ffill().fillna(1.0)
        bnh_portfolio[sym] = bnh_eq

    portfolio_eq = portfolio.mean(axis=1) * START_CAPITAL
    bnh_eq = bnh_portfolio.mean(axis=1) * START_CAPITAL

    # Stats
    end_eq = portfolio_eq.iloc[-1]
    end_bnh = bnh_eq.iloc[-1]
    total_ret = (end_eq / START_CAPITAL - 1) * 100
    bnh_ret = (end_bnh / START_CAPITAL - 1) * 100

    daily_ret = portfolio_eq.pct_change().fillna(0)
    n_days = len(portfolio_eq)
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0
    rolling_max = portfolio_eq.cummax()
    max_dd = ((portfolio_eq - rolling_max) / rolling_max).min() * 100

    n_trades = len(all_trades)
    n_wins = sum(1 for t in all_trades if t.pct_return > 0)
    avg_trade = sum(t.pct_return for t in all_trades) / n_trades * 100 if n_trades else 0
    win_rate = n_wins / n_trades * 100 if n_trades else 0

    reasons = {}
    for t in all_trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    print(f"  Starting capital:    ${START_CAPITAL:,.0f}")
    print(f"  Ending equity:       ${end_eq:,.0f}  ({total_ret:+.1f}%)")
    print(f"  Buy-and-hold parity: ${end_bnh:,.0f}  ({bnh_ret:+.1f}%)")
    print(f"  Outperformance:      {total_ret - bnh_ret:+.1f} percentage points")
    print()
    print(f"  Total trades:        {n_trades}")
    print(f"  Win rate:            {win_rate:.1f}%")
    print(f"  Avg trade return:    {avg_trade:+.2f}% (net of costs)")
    print(f"  Annualized Sharpe:   {sharpe:.2f}")
    print(f"  Max drawdown:        {max_dd:.1f}%")
    print()
    print(f"  Exit reason mix:")
    for k in ("tp", "stop", "time", "open_at_end"):
        if k in reasons:
            pct = reasons[k] / n_trades * 100
            print(f"    {k:<12} {reasons[k]:>3}  ({pct:>4.0f}%)")

    # ─── Equity curve at month boundaries ───
    print(f"\n  Portfolio equity at month-end:")
    print(f"  {'date':<12} {'portfolio':>12} {'b&h':>12} {'spread':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    monthly_pts = portfolio_eq.resample("ME").last()
    monthly_bnh = bnh_eq.resample("ME").last()
    for d in monthly_pts.index:
        p = monthly_pts.loc[d]
        b = monthly_bnh.loc[d]
        spread = p - b
        print(f"  {d.date()} {p:>11,.0f} {b:>11,.0f} {spread:>+9,.0f}")

    # ─── Per-symbol summary ───
    print(f"\n  Per-symbol performance:")
    print(f"  {'symbol':<10} {'trades':>7} {'win':>5} {'return':>8} {'b&h':>8} {'spread':>8}")
    print(f"  {'-'*10} {'-'*7} {'-'*5} {'-'*8} {'-'*8} {'-'*8}")
    for sym, s in per_sym.items():
        trades = s["trades"]
        if not trades:
            continue
        n = len(trades)
        w = sum(1 for t in trades if t.pct_return > 0)
        ret = (s["equity"].iloc[-1] - 1) * 100
        bnh = (s["df_sim"]["close"].iloc[-1] / s["df_sim"]["close"].iloc[0] - 1) * 100
        print(f"  {sym:<10} {n:>7} {w/n*100:>4.0f}% {ret:>+7.1f}% {bnh:>+7.1f}% {ret-bnh:>+7.1f}%")

    # Save equity CSV
    out_csv = Path(__file__).resolve().parent / "cache" / "hypothetical_6mo.csv"
    out_csv.parent.mkdir(exist_ok=True)
    pd.DataFrame({
        "portfolio_equity": portfolio_eq,
        "buy_and_hold": bnh_eq,
    }).to_csv(out_csv)
    print(f"\n  Equity curve saved: {out_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
