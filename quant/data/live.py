"""Live (near real-time) quote fetcher.

yfinance's `fast_info` endpoint hits Yahoo's lightweight quote API. For
US-listed equities + ETFs it's effectively real-time during market hours
(or last-trade after-hours). Crypto is 24/7 real-time. International
exchanges have ~15-min delay per Yahoo's data agreements.

No auth, no API key. Cached for 30 seconds to avoid hammering the API
and respect Yahoo's implicit rate limits.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_quote_cache: dict[str, tuple[float, dict]] = {}
QUOTE_TTL_SECONDS = 30


def get_live_quote(symbol: str, force_refresh: bool = False) -> dict | None:
    """Returns dict with price, previous_close, change, change_pct, day_high,
    day_low, currency, timestamp. Returns None on failure."""
    now = time.time()

    if not force_refresh:
        cached = _quote_cache.get(symbol)
        if cached and (now - cached[0]) < QUOTE_TTL_SECONDS:
            return cached[1]

    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        fi = ticker.fast_info
    except Exception as e:
        logger.warning("live quote fetch failed for %s: %s", symbol, e)
        return None

    def _flt(*candidates) -> float:
        """Try multiple keys; return first non-zero numeric. fast_info supports
        both attr and dict-style access in newer yfinance."""
        for key in candidates:
            try:
                v = fi[key] if hasattr(fi, "__getitem__") else getattr(fi, key, None)
            except (KeyError, AttributeError):
                v = getattr(fi, key, None)
            if v is None:
                continue
            try:
                vf = float(v)
                if vf != 0:
                    return vf
            except (TypeError, ValueError):
                continue
        return 0.0

    last = _flt("last_price", "regular_market_price", "lastPrice")
    prev_close = _flt("previous_close", "previousClose", "regular_market_previous_close")
    day_high = _flt("day_high", "dayHigh")
    day_low = _flt("day_low", "dayLow")

    if last <= 0:
        return None

    change = last - prev_close if prev_close > 0 else 0.0
    change_pct = (change / prev_close * 100) if prev_close > 0 else 0.0

    currency = "USD"
    try:
        currency = fi["currency"] if hasattr(fi, "__getitem__") else getattr(fi, "currency", "USD")
    except (KeyError, AttributeError):
        pass

    quote: dict[str, Any] = {
        "price": last,
        "previous_close": prev_close,
        "change": change,
        "change_pct": change_pct,
        "day_high": day_high,
        "day_low": day_low,
        "currency": str(currency or "USD"),
        "timestamp": now,
    }
    _quote_cache[symbol] = (now, quote)
    return quote


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    q = get_live_quote(sym, force_refresh=True)
    if q is None:
        print(f"No quote for {sym}")
        sys.exit(1)
    print(f"\n{sym}  ${q['price']:.2f} {q['currency']}")
    print(f"  Change: {q['change']:+.2f} ({q['change_pct']:+.2f}%)")
    print(f"  Day range: ${q['day_low']:.2f} – ${q['day_high']:.2f}")
    print(f"  Prev close: ${q['previous_close']:.2f}")
    print(f"  Fetched: {time.strftime('%H:%M:%S', time.localtime(q['timestamp']))}")
