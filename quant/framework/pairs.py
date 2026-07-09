"""Model 3 — Statistical Arbitrage (Pairs Trading).

Exploits temporary divergence between two historically-linked assets.
Market-neutral by construction: long one leg, short the other, profit
when the spread mean-reverts regardless of index direction.

Method (deliberately simple statistics):
  spread   = log(A) - log(B)                (classic log-ratio, beta = 1)
  zscore   = (spread - 60d mean) / 60d std
  corr     = 60-day correlation of daily returns (pair-quality check)
  half-life = ln(2) / -b from the AR(1) fit  Δspread_t = a + b·spread_{t-1}
              (how many days a divergence takes to close halfway; a spread
              that doesn't mean-revert has no edge, so we require 2-40 days)

Signals:
  z > +entry  → A rich vs B  → SHORT A / LONG B
  z < -entry  → A cheap vs B → LONG A / SHORT B
  |z| < exit  → close the trade (the gap converged)
  |z| > stop  → stop out (the relationship may have broken — Pepsi didn't
                "revert" to Coca-Cola through every regime in history)
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# Liquid, economically-linked candidates. Edit freely — the quality checks
# (correlation + half-life) decide daily which pairs are actually tradeable.
DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("KO", "PEP"),        # colas
    ("V", "MA"),          # card networks
    ("HD", "LOW"),        # home improvement
    ("XOM", "CVX"),       # oil majors
    ("GLD", "GDX"),       # gold vs gold miners
    ("QQQ", "SPY"),       # growth vs broad
    ("0700.HK", "9988.HK"),  # HK mega-tech
]


@dataclass
class PairSignal:
    a: str
    b: str
    zscore: float
    corr: float
    half_life: float
    tradeable: bool       # passed correlation + half-life quality gates
    signal: str           # "long_a_short_b" | "short_a_long_b" | "exit" | "none"
    detail: str


def _half_life(spread: pd.Series) -> float:
    """AR(1) mean-reversion half-life in days (numpy only, no statsmodels)."""
    s = spread.dropna()
    if len(s) < 30:
        return float("inf")
    lag = s.shift(1).dropna()
    delta = s.diff().dropna()
    lag, delta = lag.align(delta, join="inner")
    x = lag.to_numpy()
    y = delta.to_numpy()
    denom = np.sum((x - x.mean()) ** 2)
    if denom <= 0:
        return float("inf")
    b = float(np.sum((x - x.mean()) * (y - y.mean())) / denom)
    if b >= 0:  # no pull toward the mean at all
        return float("inf")
    return float(np.log(2) / -b)


def pair_signal(
    close_a: pd.Series,
    close_b: pd.Series,
    a: str = "A",
    b: str = "B",
    lookback: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    min_corr: float = 0.70,
    max_half_life: float = 40.0,
) -> PairSignal | None:
    """Today's signal for one pair. Returns None if data is insufficient."""
    df = pd.concat({"a": close_a, "b": close_b}, axis=1).dropna()
    if len(df) < lookback + 10:
        return None

    spread = np.log(df["a"]) - np.log(df["b"])
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std().replace(0, np.nan)
    z = float(((spread - mean) / std).iloc[-1])

    corr = float(df["a"].pct_change().rolling(lookback).corr(df["b"].pct_change()).iloc[-1])
    hl = _half_life(spread.tail(lookback * 3))

    tradeable = corr >= min_corr and 2.0 <= hl <= max_half_life
    if not tradeable:
        return PairSignal(a, b, z, corr, hl, False, "none",
                          f"pair fails quality gate (corr {corr:.2f}, half-life {hl:.0f}d)")

    if abs(z) > stop_z:
        sig, detail = "exit", f"|z|={abs(z):.1f} beyond stop {stop_z} — relationship may have broken, stand aside"
    elif z > entry_z:
        sig, detail = "short_a_long_b", f"{a} rich vs {b} by {z:.1f}σ — short {a}, long {b}, exit at |z|<{exit_z}"
    elif z < -entry_z:
        sig, detail = "long_a_short_b", f"{a} cheap vs {b} by {abs(z):.1f}σ — long {a}, short {b}, exit at |z|<{exit_z}"
    elif abs(z) < exit_z:
        sig, detail = "exit", f"spread converged (|z|={abs(z):.1f}) — close any open pair trade"
    else:
        sig, detail = "none", f"z={z:+.1f} inside the band — no action"

    return PairSignal(a, b, z, corr, hl, True, sig, detail)


def scan_pairs(
    closes: dict[str, pd.Series],
    pairs: list[tuple[str, str]] | None = None,
    **kwargs,
) -> list[PairSignal]:
    """Run pair_signal over every configured pair whose legs have data."""
    out: list[PairSignal] = []
    for a, b in (pairs or DEFAULT_PAIRS):
        if a not in closes or b not in closes:
            logger.warning("scan_pairs: missing data for (%s, %s)", a, b)
            continue
        sig = pair_signal(closes[a], closes[b], a=a, b=b, **kwargs)
        if sig is not None:
            out.append(sig)
    return out
