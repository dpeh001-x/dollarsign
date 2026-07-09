"""Model 4 — Dynamic Risk Management.

Position sizing that adapts to market stress. Three layers:

  1. Fixed-fractional risk: risk a fixed % of equity per trade (default 1%),
     with the stop distance (2.5×ATR) defining shares. A volatile stock gets
     fewer shares automatically — same dollars at risk either way.
  2. VIX scaling: when the fear gauge is above target (20), every new
     position shrinks by target/VIX. VIX 30 → 2/3 size. VIX 40 → half size.
  3. Portfolio heat cap: total open risk across ALL positions is capped
     (default 5% of equity). When the book is fully warm, new trades wait.

This is the model that keeps the other three from hurting you — the sizing
survives being wrong, which every strategy regularly is.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SizedPosition:
    symbol: str
    side: str            # "long" | "short"
    entry: float
    stop: float
    shares: int
    notional: float
    risk_dollars: float  # loss if the stop is hit
    vix_scalar: float
    note: str = ""


def vix_scalar(vix_level: float | None, vix_target: float = 20.0, floor: float = 0.25) -> float:
    """Size multiplier from volatility: 1.0 in calm markets, shrinking as
    VIX rises above target, never below `floor` (avoid zero-size dust)."""
    if vix_level is None or vix_level <= 0 or vix_level <= vix_target:
        return 1.0
    return max(vix_target / vix_level, floor)


def atr_stop(entry: float, atr: float, side: str, mult: float = 2.5) -> float:
    """Volatility-adjusted stop: 2.5×ATR against the position."""
    return entry - mult * atr if side == "long" else entry + mult * atr


def size_position(
    symbol: str,
    side: str,
    entry: float,
    atr: float,
    equity: float,
    risk_pct: float = 1.0,
    vix_level: float | None = None,
    vix_target: float = 20.0,
    stop_mult: float = 2.5,
    max_position_pct: float = 20.0,
    size_scale: float = 1.0,
) -> SizedPosition | None:
    """Full sizing pipeline for one trade. Returns None if the position
    rounds to zero shares (account too small for this symbol's volatility).

    size_scale lets the framework halve size in 'neutral' regimes without
    touching the risk math.
    """
    if entry <= 0 or atr <= 0 or equity <= 0:
        return None

    stop = atr_stop(entry, atr, side, stop_mult)
    stop_dist = abs(entry - stop)

    scalar = vix_scalar(vix_level, vix_target)
    risk_dollars = equity * (risk_pct / 100.0) * scalar * size_scale

    shares = int(risk_dollars / stop_dist)
    if shares <= 0:
        return None

    # Notional cap: no single name above max_position_pct of equity,
    # regardless of how tight its stop is.
    max_notional = equity * (max_position_pct / 100.0)
    if shares * entry > max_notional:
        shares = int(max_notional / entry)
        if shares <= 0:
            return None
        risk_dollars = shares * stop_dist

    note = ""
    if scalar < 1.0:
        note = f"size cut to {scalar:.0%} (VIX {vix_level:.0f} > target {vix_target:.0f})"

    return SizedPosition(
        symbol=symbol, side=side, entry=round(entry, 2), stop=round(stop, 2),
        shares=shares, notional=round(shares * entry, 2),
        risk_dollars=round(shares * stop_dist, 2), vix_scalar=scalar, note=note,
    )


def apply_heat_cap(
    positions: list[SizedPosition],
    equity: float,
    max_heat_pct: float = 5.0,
) -> tuple[list[SizedPosition], list[SizedPosition]]:
    """Enforce the portfolio heat cap: accept positions in order until total
    at-risk dollars would exceed max_heat_pct of equity; defer the rest.

    Returns (accepted, deferred). Callers pass positions in priority order
    (strongest signal first) so the best ideas get the risk budget.
    """
    budget = equity * (max_heat_pct / 100.0)
    accepted: list[SizedPosition] = []
    deferred: list[SizedPosition] = []
    used = 0.0
    for p in positions:
        if used + p.risk_dollars <= budget:
            accepted.append(p)
            used += p.risk_dollars
        else:
            deferred.append(p)
    return accepted, deferred
