"""Analyst consensus data from yfinance.

Wall Street analyst price targets are strong short-term signal:
they aggregate institutional research and tend to lead price by weeks.
We use them for two things:
  1. Consensus gating — only signal BUY/SELL when the ML model
     agrees with the analyst consensus direction (raises accuracy).
  2. Target price — the analyst mean target is shown as the trade's
     objective (institutional wisdom, not an arbitrary ATR multiple).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AnalystView:
    mean_target: float | None      # average of all analyst 12-mo price targets
    high_target: float | None      # highest analyst target
    low_target: float | None       # lowest analyst target
    num_analysts: int              # how many analysts contributed
    recommendation: str | None     # 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'
    current_price: float | None    # spot price from yfinance
    upside_pct: float | None       # (mean_target / current_price - 1) * 100

    @property
    def direction(self) -> str:
        """Consensus direction: 'up', 'down', or 'flat' based on upside_pct."""
        if self.upside_pct is None:
            return "flat"
        if self.upside_pct > 3.0:
            return "up"
        if self.upside_pct < -3.0:
            return "down"
        return "flat"

    @property
    def is_high_conviction(self) -> bool:
        """>10 analysts covering the name → institutional consensus is meaningful."""
        return self.num_analysts >= 10


def get_analyst_view(symbol: str) -> AnalystView | None:
    """Return AnalystView for a symbol, or None if unavailable.

    Uses yfinance's .info endpoint which aggregates analyst estimates
    from major sell-side firms (Goldman, JPM, Morgan Stanley, etc.).
    """
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        info = t.info or {}
    except Exception as e:
        logger.warning("analyst fetch failed for %s: %s", symbol, e)
        return None

    mean = info.get("targetMeanPrice")
    high = info.get("targetHighPrice")
    low = info.get("targetLowPrice")
    n = info.get("numberOfAnalystOpinions") or 0
    rec = info.get("recommendationKey")
    price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")

    upside = None
    if mean is not None and price:
        try:
            upside = (float(mean) / float(price) - 1.0) * 100.0
        except (TypeError, ValueError, ZeroDivisionError):
            upside = None

    return AnalystView(
        mean_target=float(mean) if mean is not None else None,
        high_target=float(high) if high is not None else None,
        low_target=float(low) if low is not None else None,
        num_analysts=int(n) if n else 0,
        recommendation=str(rec) if rec else None,
        current_price=float(price) if price else None,
        upside_pct=upside,
    )
