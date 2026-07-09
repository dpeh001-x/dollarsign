"""Model 1 — Market Regime Filter.

Checks the overall market state BEFORE any strategy fires. Uses simple,
transparent statistics on a benchmark index (SPY by default, ^HSI for a
Hong Kong book):

  - 50-day and 200-day moving averages (trend)
  - 50-day MA slope (is the trend improving or rolling over?)
  - 20-day realized volatility percentile vs the last year (stress)

States:
  risk_on   — benchmark above a rising 50d MA, above 200d MA, calm vol.
              Momentum longs allowed at full size.
  neutral   — chop/consolidation. Only mean-reversion entries, half size.
  risk_off  — benchmark below 200d MA, or below 50d MA with vol in the
              top ~15% of the year. NO new longs — this is the switch that
              stops "buying the dip" into a bear market.

Market-neutral sleeves (pairs) ignore the filter by design.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class RegimeState:
    state: str          # "risk_on" | "neutral" | "risk_off"
    detail: str         # one-line human explanation
    benchmark: str
    close: float
    ma50: float
    ma200: float
    ma50_slope: float   # 10-day change of the 50d MA, as pct of price
    vol_pctile: float   # 20d realized vol percentile vs trailing year (0-1)

    @property
    def allow_new_longs(self) -> bool:
        return self.state != "risk_off"

    @property
    def full_size(self) -> bool:
        return self.state == "risk_on"


def classify_regime(bench_close: pd.Series, benchmark: str = "SPY") -> RegimeState:
    """Classify the market regime from a benchmark's daily close series."""
    close = bench_close.dropna()
    if len(close) < 260:
        raise ValueError(f"Need >= 260 daily closes for regime filter; have {len(close)}")

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    ma50_slope = (ma50 - ma50.shift(10)) / close

    ret = close.pct_change()
    vol20 = ret.rolling(20).std()
    vol_pctile = vol20.rolling(252).rank(pct=True)

    c, m50, m200 = float(close.iloc[-1]), float(ma50.iloc[-1]), float(ma200.iloc[-1])
    slope, vp = float(ma50_slope.iloc[-1]), float(vol_pctile.iloc[-1])

    if c < m200 or (c < m50 and vp > 0.85):
        state, detail = "risk_off", (
            f"{benchmark} below its 200-day average" if c < m200
            else f"{benchmark} below its 50-day average with volatility in the top {100 - vp*100:.0f}% of the year"
        )
    elif c > m50 > m200 and slope > 0:
        state, detail = "risk_on", f"{benchmark} above a rising 50-day average, above the 200-day"
    else:
        state, detail = "neutral", f"{benchmark} consolidating — no clean trend"

    return RegimeState(
        state=state, detail=detail, benchmark=benchmark,
        close=c, ma50=m50, ma200=m200, ma50_slope=slope, vol_pctile=vp,
    )


def market_regime(benchmark: str = "SPY", start: str | None = None) -> RegimeState:
    """Fetch the benchmark and classify today's regime."""
    from datetime import date, timedelta
    from data import sources

    if start is None:
        start = (date.today() - timedelta(days=550)).isoformat()
    bars = sources.get_bars(benchmark, start, use_cache=True)
    return classify_regime(bars["close"], benchmark=benchmark)
