"""Backtest the multi-factor framework as one portfolio.

Usage:
    python backtest_portfolio_cli.py --equity 10000
    python backtest_portfolio_cli.py --equity 25000 --start 2022-07-01 --risk 1.0
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import sources  # noqa: E402
from framework.backtest_portfolio import run_portfolio_backtest, BTConfig  # noqa: E402
from framework.framework import DEFAULT_UNIVERSE  # noqa: E402
from framework.pairs import DEFAULT_PAIRS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Portfolio backtest of the 4-model framework")
    ap.add_argument("--equity", type=float, default=10_000)
    ap.add_argument("--risk", type=float, default=1.0)
    ap.add_argument("--start", type=str, default=(date.today() - timedelta(days=4 * 365)).isoformat())
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    cfg = BTConfig(equity=args.equity, risk_pct=args.risk, top_n=args.top)

    symbols = sorted(set(cfg.universe) | {s for p in cfg.pairs for s in p} | {cfg.benchmark})
    print(f"Loading {len(symbols)} symbols from {args.start}...")
    bars = {}
    for sym in symbols:
        try:
            bars[sym] = sources.get_bars(sym, args.start, end=args.end, use_cache=True)
        except Exception as e:
            print(f"  skip {sym}: {e}")

    vix = None
    try:
        vix = sources.get_bars("^VIX", args.start, end=args.end, use_cache=True)["close"]
    except Exception as e:
        print(f"  VIX unavailable ({e}) — sizing without volatility scaling")

    print(f"Running backtest on {len(bars)} symbols...")
    r = run_portfolio_backtest(bars, vix_close=vix, cfg=cfg)

    print(f"\n{'='*64}")
    print(f"  PORTFOLIO BACKTEST · {r['start']} → {r['end']}")
    print(f"{'='*64}")
    print(f"  Initial equity:    ${r['initial_equity']:>12,.0f}")
    print(f"  Final equity:      ${r['final_equity']:>12,.2f}")
    print(f"  Total return:      {r['total_return']*100:>+12.1f}%")
    print(f"  CAGR:              {r['cagr']*100:>+12.1f}%")
    print(f"  Sharpe:            {r['sharpe']:>12.2f}")
    print(f"  Max drawdown:      {r['max_drawdown']*100:>12.1f}%")
    print(f"  Trades:            {r['n_trades']:>12d}   win rate {r['win_rate']*100:.0f}%")
    print(f"  {r.get('benchmark', 'SPY')} buy & hold:   {r['benchmark_return']*100:>+12.1f}%  (same window)")

    print(f"\n  Regime days: " + ", ".join(f"{k} {v}" for k, v in r["regime_days"].items()))
    if r["sleeve_pnl"]:
        print(f"\n  Sleeve P&L:")
        for sleeve, d in r["sleeve_pnl"].items():
            print(f"    {sleeve:<10} {d['count']:>4.0f} trades   ${d['sum']:>+10,.2f}")

    tdf = r["trades"]
    if len(tdf):
        print(f"\n  Worst 3 trades:")
        for _, t in tdf.nsmallest(3, "pnl").iterrows():
            print(f"    {t['symbol']:<8} {t['sleeve']:<9} {t['pnl']:>+9.2f}  ({t['reason']}, {t['exit_date']})")
        print(f"  Best 3 trades:")
        for _, t in tdf.nlargest(3, "pnl").iterrows():
            print(f"    {t['symbol']:<8} {t['sleeve']:<9} {t['pnl']:>+9.2f}  ({t['reason']}, {t['exit_date']})")

    print("\n  Execution model: next-close fills, close-checked stops, 2bps/side costs.")
    print("  Past performance does not guarantee future results.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
