"""End-to-end CLI: fetch real OHLCV -> indicators -> regime -> backtest -> report.

Usage:
    python run.py --symbol AAPL --start 2022-01-01
    python run.py --symbol SPY --start 2020-01-01 --strategy buy_hold
    python run.py --symbol BTC-USD --start 2023-01-01 --strategy regime_breakout
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("quant.run")

from data import sources  # noqa: E402
import indicators  # noqa: E402
import regime  # noqa: E402
import backtest  # noqa: E402
import strategies  # noqa: E402

STRATEGIES = {
    "buy_hold": lambda df: __import__("pandas").Series(1.0, index=df.index),
    "regime_breakout": strategies.regime_aware_breakout,
    "rsi_meanrev": strategies.rsi_mean_reversion,
    "ma_cross": strategies.ma_crossover,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Quant pipeline: data -> indicators -> regime -> backtest")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--timeframe", default="1Day", choices=["1Day", "1Hour"])
    parser.add_argument("--strategy", default="regime_breakout", choices=list(STRATEGIES))
    parser.add_argument("--fee_bps", type=float, default=1.0)
    parser.add_argument("--slippage_bps", type=float, default=1.0)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--source", default="auto", choices=["auto", "alpaca", "polygon", "yfinance"])
    args = parser.parse_args()

    end = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"\n{'='*60}\n  {args.symbol}  {args.start} -> {end}  [{args.timeframe}]\n{'='*60}")

    df = sources.get_bars(
        symbol=args.symbol,
        start=args.start,
        end=end,
        timeframe=args.timeframe,
        use_cache=not args.no_cache,
        prefer=args.source,
    )
    print(f"\nData source: {df.attrs.get('source', '?')}  ·  Bars: {len(df):,}")
    print(f"Date range:  {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Price range: ${df['close'].min():.2f} -> ${df['close'].max():.2f}  (last: ${df['close'].iloc[-1]:.2f})")

    print("\n[1/3] Computing indicators...")
    df = indicators.attach_all(df)

    print("[2/3] Classifying regimes...")
    df = regime.label(df)
    stats = regime.regime_stats(df)
    print("\nRegime distribution + per-regime forward-1-bar return:")
    print(stats.to_string())

    print(f"\n[3/3] Backtesting strategy: {args.strategy}")
    strat_fn = STRATEGIES[args.strategy]
    positions = strat_fn(df)
    bars_per_year = 252 if args.timeframe == "1Day" else int(252 * 6.5)
    result = backtest.run(
        df,
        positions,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        bars_per_year=bars_per_year,
    )

    print("\nStrategy performance:")
    print(result.summary())

    if args.strategy != "buy_hold":
        bh = backtest.buy_and_hold(df, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps, bars_per_year=bars_per_year)
        print("\nBuy-and-hold benchmark:")
        print(bh.summary())

    if len(result.trades) > 0:
        print(f"\nLast 5 trades:")
        cols = ["entry_time", "exit_time", "side", "entry_price", "exit_price", "net_pct", "bars_held"]
        print(result.trades[cols].tail(5).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
