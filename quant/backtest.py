"""Vectorized backtest engine for long/short signal strategies.

Strategy contract: a function (df: DataFrame) -> Series of positions in
{-1, 0, +1} (or fractional in [-1, 1] for sizing) indexed like df. The
position at bar t is *entered at the close of t* and held into t+1; PnL
for t+1 is therefore position[t] * return[t+1]. This avoids look-ahead bias.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float]

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"Bars:           {int(m['bars']):>10,d}",
            f"Trades:         {int(m['trades']):>10,d}",
            f"Win rate:       {m['win_rate']:>10.1%}",
            f"Total return:   {m['total_return']:>10.1%}",
            f"CAGR:           {m['cagr']:>10.1%}",
            f"Sharpe:         {m['sharpe']:>10.2f}",
            f"Max drawdown:   {m['max_dd']:>10.1%}",
            f"Calmar:         {m['calmar']:>10.2f}",
            f"Avg trade:      {m['avg_trade_bps']:>10.1f} bps",
        ]
        return "\n".join(lines)


def run(
    df: pd.DataFrame,
    positions: pd.Series,
    fee_bps: float = 1.0,
    slippage_bps: float = 1.0,
    initial_capital: float = 100_000.0,
    bars_per_year: int = 252,
) -> BacktestResult:
    """Run a vectorized backtest.

    Args:
        df: OHLCV frame (lowercase cols). Must contain 'close'.
        positions: series in [-1, 1] aligned to df.index. Entered at bar's close,
            held into next bar.
        fee_bps: per-side trading fee in basis points (1 bp = 0.01%).
        slippage_bps: per-side slippage assumption.
        initial_capital: starting equity for $-denominated stats.
        bars_per_year: 252 for daily, 252*6.5 for hourly, etc.
    """
    pos = positions.reindex(df.index).fillna(0).clip(-1, 1)

    # Per-bar return (close-to-close, log-equivalent for compounding)
    bar_ret = df["close"].pct_change().fillna(0)

    # Position from bar t earns return at bar t+1 — shift positions forward
    held = pos.shift(1).fillna(0)
    gross_ret = held * bar_ret

    # Trading costs hit on position changes
    pos_change = pos.diff().abs().fillna(pos.iloc[0])  # entry costs at first bar
    cost_per_change = (fee_bps + slippage_bps) / 10_000
    costs = pos_change * cost_per_change
    net_ret = gross_ret - costs

    equity = (1 + net_ret).cumprod() * initial_capital

    # Trade ledger: track entries/exits
    trades = _extract_trades(pos, df["close"], fee_bps + slippage_bps)
    metrics = _compute_metrics(net_ret, equity, trades, bars_per_year)

    return BacktestResult(
        equity=equity,
        returns=net_ret,
        positions=pos,
        trades=trades,
        metrics=metrics,
    )


def _extract_trades(pos: pd.Series, close: pd.Series, total_cost_bps: float) -> pd.DataFrame:
    """Walk the position series to extract discrete round-trip trades."""
    trades: list[dict] = []
    in_pos = 0.0
    entry_price = np.nan
    entry_time = None

    for ts, p in pos.items():
        if p != in_pos:
            # Close existing position
            if in_pos != 0 and entry_time is not None:
                exit_price = close.loc[ts]
                gross_pct = (exit_price / entry_price - 1) * np.sign(in_pos)
                net_pct = gross_pct - 2 * (total_cost_bps / 10_000)  # entry + exit
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "side": "long" if in_pos > 0 else "short",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_pct": gross_pct,
                    "net_pct": net_pct,
                    "bars_held": (close.index.get_loc(ts) - close.index.get_loc(entry_time)),
                })
            # Open new position
            if p != 0:
                entry_price = close.loc[ts]
                entry_time = ts
            in_pos = p

    return pd.DataFrame(trades)


def _compute_metrics(net_ret: pd.Series, equity: pd.Series, trades: pd.DataFrame, bars_per_year: int) -> dict[str, float]:
    bars = len(net_ret)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1 if len(equity) else 0.0
    years = bars / bars_per_year if bars_per_year else 1.0
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else 0.0

    # Sharpe (assuming 0 risk-free)
    if net_ret.std() > 0:
        sharpe = (net_ret.mean() / net_ret.std()) * np.sqrt(bars_per_year)
    else:
        sharpe = 0.0

    # Max drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min() if len(drawdown) else 0.0
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    if len(trades) > 0:
        win_rate = (trades["net_pct"] > 0).mean()
        avg_trade_bps = trades["net_pct"].mean() * 10_000
    else:
        win_rate = 0.0
        avg_trade_bps = 0.0

    return {
        "bars": float(bars),
        "trades": float(len(trades)),
        "win_rate": float(win_rate),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "calmar": float(calmar),
        "avg_trade_bps": float(avg_trade_bps),
    }


def buy_and_hold(df: pd.DataFrame, **kwargs) -> BacktestResult:
    """Reference benchmark: always long."""
    pos = pd.Series(1.0, index=df.index)
    return run(df, pos, **kwargs)
