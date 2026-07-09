"""Daily trade plan from the multi-factor framework.

Usage:
    python framework_cli.py                          # defaults: $10k, 1% risk
    python framework_cli.py --equity 25000 --risk 1.5 --shorts
    python framework_cli.py --universe AAPL,NVDA,PLTR,0700.HK --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from framework import daily_plan, FrameworkConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-factor momentum + mean-reversion daily plan")
    ap.add_argument("--equity", type=float, default=10_000, help="account equity")
    ap.add_argument("--risk", type=float, default=1.0, help="%% of equity risked per trade")
    ap.add_argument("--heat", type=float, default=5.0, help="max total open risk %% of equity")
    ap.add_argument("--top", type=int, default=5, help="momentum longs")
    ap.add_argument("--shorts", action="store_true", help="enable momentum shorts in risk-off")
    ap.add_argument("--universe", type=str, default="", help="comma-separated symbols (default: built-in basket)")
    ap.add_argument("--benchmark", type=str, default="SPY")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = ap.parse_args()

    cfg = FrameworkConfig(
        account_equity=args.equity, risk_pct=args.risk, max_heat_pct=args.heat,
        top_n=args.top, allow_shorts=args.shorts, benchmark=args.benchmark,
    )
    if args.universe:
        cfg.universe = [s.strip().upper() for s in args.universe.split(",") if s.strip()]

    plan = daily_plan(cfg)

    if args.json:
        print(json.dumps(plan, indent=2, default=str))
        return 0

    r = plan["regime"]
    state_banner = {"risk_on": "RISK ON", "neutral": "NEUTRAL", "risk_off": "RISK OFF — no new longs"}
    print(f"\n{'='*72}")
    print(f"  DAILY PLAN · {plan['as_of']} · equity ${plan['config']['equity']:,.0f}")
    print(f"{'='*72}")
    print(f"\n[1] REGIME ({r['benchmark']}): {state_banner.get(r['state'], r['state'])}")
    print(f"    {r['detail']}")
    if plan["vix"] is not None:
        print(f"    VIX {plan['vix']:.1f}")

    print(f"\n[2] CROSS-SECTIONAL MOMENTUM (top 10 of {len(plan['ranks'])} ranked)")
    if plan["ranks"]:
        print(f"    {'rank':>4} {'symbol':<9} {'score':>7} {'63d':>7} {'126d':>7} {'RSI':>5}  flag")
        for row in plan["ranks"][:10]:
            print(f"    {row['rank']:>4} {row['symbol']:<9} {row['score']*100:>+6.1f}% "
                  f"{row['ret_63']*100:>+6.1f}% {row['ret_126']*100:>+6.1f}% "
                  f"{row['rsi_14']:>5.0f}  {row['mr_flag']}")
    print(f"    momentum buys: {', '.join(plan['momentum_buys']) or '(none — regime gated or no positive scores)'}")
    print(f"    dip buys:      {', '.join(plan['dip_buys']) or '(none oversold in uptrends)'}")
    if plan["shorts"]:
        print(f"    shorts:        {', '.join(plan['shorts'])}")

    print(f"\n[3] PAIRS (statistical arbitrage)")
    for p in plan["pairs"]:
        mark = "→" if p["signal"] in ("long_a_short_b", "short_a_long_b") else " "
        print(f"  {mark} {p['a']}/{p['b']}: z={p['zscore']:+.2f} corr={p['corr']:.2f} "
              f"HL={p['half_life']:.0f}d — {p['detail']}")

    print(f"\n[4] SIZED TRADES (risk {plan['config']['risk_pct']}%/trade, "
          f"heat cap {plan['config']['max_heat_pct']}%, total at risk ${plan['open_risk_dollars']:,.0f})")
    if plan["trades"]:
        print(f"    {'symbol':<9} {'side':<6} {'shares':>7} {'entry':>9} {'stop':>9} {'risk $':>8}  note")
        for t in plan["trades"]:
            print(f"    {t['symbol']:<9} {t['side']:<6} {t['shares']:>7} {t['entry']:>9,.2f} "
                  f"{t['stop']:>9,.2f} {t['risk_dollars']:>8,.0f}  {t['note']}")
    else:
        print("    (no trades today — cash is a position)")
    if plan["deferred"]:
        print(f"    deferred by heat cap: {', '.join(t['symbol'] for t in plan['deferred'])}")

    print("\n  Every trade: place the stop when you place the order. Exit pairs at |z|<0.5.")
    print("  Not financial advice; backtest before risking real money.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
