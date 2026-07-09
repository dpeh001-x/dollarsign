"""Portfolio backtest of the multi-factor framework.

Simulates one account running all four models together, day by day:

  - Regime filter (vectorized over the benchmark) gates the sleeves.
  - Momentum sleeve: rebalanced every 21 bars in risk_on — long the top-N
    ranked names (RSI-capped), closed whenever the regime leaves risk_on.
  - Dip sleeve: RSI<30 above the 200d MA, max 3 open, exits on RSI>55,
    a 2.5xATR stop, a 20-bar time stop, or regime turning risk_off.
  - Pairs sleeve: z-score entries/exits on quality-gated pairs, dollar
    neutral (long leg + short leg), max 2 concurrent pairs, 40-bar time stop.
  - Risk: 1% fixed-fractional per stock trade off the ATR stop, VIX-scaled,
    5% portfolio heat cap; pair legs at 5% notional each, VIX-scaled.

Execution model (deliberately conservative and simple):
  - Signals computed on day t's close, filled at day t+1's close.
  - Stops evaluated on closes (no intraday fills), exit at next close.
  - Costs: 2 bps per side (fee + slippage) on traded notional.
  - Non-US symbols are marked/filled at their last available close on the
    benchmark calendar (HK holidays -> previous close).

Not investment advice; an approximation with honest, known simplifications.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import indicators  # noqa: E402
from framework import risk as risk_mod  # noqa: E402
from framework.pairs import DEFAULT_PAIRS, _half_life  # noqa: E402
from framework.framework import DEFAULT_UNIVERSE  # noqa: E402

logger = logging.getLogger(__name__)

COST = 2e-4  # 2 bps per side


@dataclass
class BTConfig:
    equity: float = 10_000.0
    risk_pct: float = 1.0
    max_heat_pct: float = 5.0
    vix_target: float = 20.0
    top_n: int = 5
    dip_n: int = 3
    max_pairs: int = 2
    pair_leg_pct: float = 5.0      # notional per pair leg, % of equity
    rebalance_every: int = 21
    universe: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    pairs: list[tuple[str, str]] = field(default_factory=lambda: list(DEFAULT_PAIRS))
    benchmark: str = "SPY"


@dataclass
class Position:
    symbol: str
    side: str            # "long" | "short"
    shares: int
    entry: float
    stop: float | None
    sleeve: str          # "momentum" | "dip" | "pair"
    opened_i: int
    pair_id: str = ""
    risk_dollars: float = 0.0


def _regime_series(close: pd.Series) -> pd.Series:
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    slope = (ma50 - ma50.shift(10)) / close
    volp = close.pct_change().rolling(20).std().rolling(252).rank(pct=True)
    risk_off = (close < ma200) | ((close < ma50) & (volp > 0.85))
    risk_on = (close > ma50) & (ma50 > ma200) & (slope > 0) & ~risk_off
    return pd.Series(np.where(risk_off, "risk_off", np.where(risk_on, "risk_on", "neutral")),
                     index=close.index)


def run_portfolio_backtest(
    bars: dict[str, pd.DataFrame],
    vix_close: pd.Series | None = None,
    cfg: BTConfig | None = None,
) -> dict:
    cfg = cfg or BTConfig()
    if cfg.benchmark not in bars:
        raise ValueError(f"benchmark {cfg.benchmark} missing from bars")

    cal = bars[cfg.benchmark].index  # trading calendar = benchmark days
    syms = [s for s in set(cfg.universe) | {x for p in cfg.pairs for x in p} | {cfg.benchmark} if s in bars]

    close = pd.DataFrame({s: bars[s]["close"].reindex(cal).ffill() for s in syms})
    atr = pd.DataFrame({s: indicators.atr(bars[s], 14).reindex(cal).ffill() for s in syms})
    rsi = pd.DataFrame({s: indicators.rsi(bars[s]["close"], 14).reindex(cal).ffill() for s in syms})
    ma200 = close.rolling(200).mean()
    c5 = close.shift(5)
    score = 0.5 * (c5 / close.shift(68) - 1) + 0.5 * (c5 / close.shift(131) - 1)

    regime = _regime_series(bars[cfg.benchmark]["close"]).reindex(cal).ffill()
    vix = vix_close.reindex(cal).ffill() if vix_close is not None else pd.Series(np.nan, index=cal)

    # pair spreads + rolling stats
    spreads = {}
    for a, b in cfg.pairs:
        if a in close.columns and b in close.columns:
            sp = np.log(close[a]) - np.log(close[b])
            spreads[(a, b)] = {
                "z": (sp - sp.rolling(60).mean()) / sp.rolling(60).std().replace(0, np.nan),
                "corr": close[a].pct_change().rolling(60).corr(close[b].pct_change()),
                "spread": sp,
            }

    n = len(cal)
    warmup = 265
    cash = cfg.equity
    positions: list[Position] = []
    equity_curve = pd.Series(np.nan, index=cal)
    trades: list[dict] = []
    pending: list[tuple] = []   # orders decided at t, filled at t+1 close

    def mark_equity(i: int) -> float:
        val = cash
        for p in positions:
            px = close[p.symbol].iloc[i]
            val += p.shares * px if p.side == "long" else -p.shares * px
        return float(val)

    def stock_heat() -> float:
        return sum(p.risk_dollars for p in positions if p.sleeve in ("momentum", "dip"))

    def fill(i: int, order: tuple):
        nonlocal cash
        kind = order[0]
        if kind == "open":
            _, sym, side, shares, stop, sleeve, pair_id, risk_d = order
            px = float(close[sym].iloc[i])
            notional = shares * px
            cash += (-notional if side == "long" else notional)
            cash -= notional * COST
            positions.append(Position(sym, side, shares, px, stop, sleeve, i, pair_id, risk_d))
        else:  # close
            _, pos, reason = order
            if pos not in positions:
                return
            px = float(close[pos.symbol].iloc[i])
            notional = pos.shares * px
            cash += (notional if pos.side == "long" else -notional)
            cash -= notional * COST
            pnl = (px - pos.entry) * pos.shares * (1 if pos.side == "long" else -1)
            trades.append({
                "symbol": pos.symbol, "sleeve": pos.sleeve, "side": pos.side,
                "entry": pos.entry, "exit": px, "pnl": round(pnl, 2),
                "bars_held": i - pos.opened_i, "reason": reason,
                "exit_date": str(cal[i].date()),
            })
            positions.remove(pos)

    open_pair_ids: set[str] = set()

    for i in range(warmup, n - 1):
        # 1. fill yesterday's orders at today's close
        for order in pending:
            fill(i, order)
        pending = []
        open_pair_ids = {p.pair_id for p in positions if p.sleeve == "pair"}

        eq = mark_equity(i)
        equity_curve.iloc[i] = eq
        state = regime.iloc[i]
        vx = float(vix.iloc[i]) if not np.isnan(vix.iloc[i]) else None
        scale = 1.0 if state == "risk_on" else 0.5

        # 2. exits
        for p in list(positions):
            sym = p.symbol
            px = close[sym].iloc[i]
            if p.sleeve in ("momentum", "dip"):
                if state == "risk_off":
                    pending.append(("close", p, "regime risk_off"))
                elif p.stop and ((p.side == "long" and px < p.stop) or (p.side == "short" and px > p.stop)):
                    pending.append(("close", p, "stop"))
                elif p.sleeve == "dip" and (rsi[sym].iloc[i] > 55 or i - p.opened_i >= 20):
                    pending.append(("close", p, "dip exit (RSI>55 or 20 bars)"))
                elif p.sleeve == "momentum" and state != "risk_on":
                    pending.append(("close", p, "regime left risk_on"))

        for (a, b), d in spreads.items():
            pid = f"{a}/{b}"
            if pid not in open_pair_ids:
                continue
            z = d["z"].iloc[i]
            legs = [p for p in positions if p.pair_id == pid]
            oldest = min((p.opened_i for p in legs), default=i)
            if np.isnan(z) or abs(z) < 0.5 or abs(z) > 3.5 or i - oldest >= 40:
                why = "pair stop |z|>3.5" if abs(z) > 3.5 else ("pair converged" if abs(z) < 0.5 else "pair time stop")
                for p in legs:
                    pending.append(("close", p, why))

        closing = {id(o[1]) for o in pending if o[0] == "close"}
        live = [p for p in positions if id(p) not in closing]

        # 3. momentum rebalance
        if state == "risk_on" and (i - warmup) % cfg.rebalance_every == 0:
            row = pd.DataFrame({
                "score": score.iloc[i], "rsi_14": rsi.iloc[i],
                "above_ma200": close.iloc[i] > ma200.iloc[i],
            }).dropna()
            row = row[row.index.isin(cfg.universe)].sort_values("score", ascending=False)
            row["rank"] = np.arange(1, len(row) + 1)
            eligible = row[(row["rank"] <= cfg.top_n * 2) & (row["rsi_14"] <= 75) & (row["score"] > 0)]
            target = list(eligible.index[:cfg.top_n])

            for p in live:
                if p.sleeve == "momentum" and p.symbol not in target:
                    pending.append(("close", p, "rebalance"))
            held = {p.symbol for p in live if p.sleeve == "momentum"}
            for sym in target:
                if sym in held:
                    continue
                sized = risk_mod.size_position(sym, "long", float(close[sym].iloc[i]), float(atr[sym].iloc[i]),
                                               eq, cfg.risk_pct, vx, cfg.vix_target, size_scale=scale)
                if sized and stock_heat() + sized.risk_dollars <= eq * cfg.max_heat_pct / 100:
                    pending.append(("open", sym, "long", sized.shares, sized.stop, "momentum", "", sized.risk_dollars))

        # 4. dip entries
        if state != "risk_off":
            n_dips = sum(1 for p in live if p.sleeve == "dip")
            if n_dips < cfg.dip_n:
                flags = (rsi.iloc[i] < 30) & (close.iloc[i] > ma200.iloc[i])
                held = {p.symbol for p in live}
                for sym in [s for s in cfg.universe if s in flags.index and flags[s] and s not in held][: cfg.dip_n - n_dips]:
                    dip_scale = scale * (0.5 if state == "neutral" else 1.0)
                    sized = risk_mod.size_position(sym, "long", float(close[sym].iloc[i]), float(atr[sym].iloc[i]),
                                                   eq, cfg.risk_pct, vx, cfg.vix_target, size_scale=dip_scale)
                    if sized and stock_heat() + sized.risk_dollars <= eq * cfg.max_heat_pct / 100:
                        pending.append(("open", sym, "long", sized.shares, sized.stop, "dip", "", sized.risk_dollars))

        # 5. pair entries
        n_pairs_open = len({p.pair_id for p in live if p.sleeve == "pair"})
        for (a, b), d in spreads.items():
            pid = f"{a}/{b}"
            if pid in open_pair_ids or n_pairs_open >= cfg.max_pairs:
                continue
            z, corr = d["z"].iloc[i], d["corr"].iloc[i]
            if np.isnan(z) or np.isnan(corr) or corr < 0.70 or not (2.0 < abs(z) <= 3.5):
                continue
            hl = _half_life(d["spread"].iloc[max(0, i - 180): i + 1])
            if not (2.0 <= hl <= 40.0):
                continue
            leg_notional = eq * cfg.pair_leg_pct / 100 * risk_mod.vix_scalar(vx, cfg.vix_target)
            long_leg, short_leg = (b, a) if z > 0 else (a, b)
            sh_l = int(leg_notional / close[long_leg].iloc[i])
            sh_s = int(leg_notional / close[short_leg].iloc[i])
            if sh_l > 0 and sh_s > 0:
                pending.append(("open", long_leg, "long", sh_l, None, "pair", pid, 0.0))
                pending.append(("open", short_leg, "short", sh_s, None, "pair", pid, 0.0))
                n_pairs_open += 1

    # liquidate at the end
    for p in list(positions):
        fill(n - 1, ("close", p, "end of backtest"))
    equity_curve.iloc[n - 1] = cash

    eqc = equity_curve.dropna()
    ret = eqc.pct_change().dropna()
    years = len(eqc) / 252
    total = float(eqc.iloc[-1] / cfg.equity - 1)
    dd = (eqc / eqc.cummax() - 1).min()
    tdf = pd.DataFrame(trades)
    bench = close[cfg.benchmark].iloc[warmup:n]
    sleeve_pnl = tdf.groupby("sleeve")["pnl"].agg(["sum", "count"]) if len(tdf) else pd.DataFrame()

    return {
        "start": str(eqc.index[0].date()), "end": str(eqc.index[-1].date()),
        "initial_equity": cfg.equity, "final_equity": round(float(eqc.iloc[-1]), 2),
        "total_return": total,
        "cagr": (1 + total) ** (1 / years) - 1 if years > 0 else 0.0,
        "sharpe": float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0,
        "max_drawdown": float(dd),
        "n_trades": len(tdf),
        "win_rate": float((tdf["pnl"] > 0).mean()) if len(tdf) else 0.0,
        "benchmark_return": float(bench.iloc[-1] / bench.iloc[0] - 1),
        "sleeve_pnl": sleeve_pnl.to_dict(orient="index") if len(sleeve_pnl) else {},
        "regime_days": regime.iloc[warmup:].value_counts().to_dict(),
        "equity_curve": eqc,
        "trades": tdf,
    }
