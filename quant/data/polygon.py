"""Polygon.io stock/crypto aggregates adapter.

Free tier: 5 calls/min, 2 years of history. Use only for daily bars or
small backfills — burst much past 5 req/min and you'll get 429s.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

logger = logging.getLogger(__name__)

POLYGON_API = "https://api.polygon.io"

TIMEFRAME_MAP = {
    "1Day": (1, "day"),
    "1Hour": (1, "hour"),
    "1Min": (1, "minute"),
    "5Min": (5, "minute"),
}


def fetch_bars(symbol: str, start: str, end: str, timeframe: str = "1Day") -> pd.DataFrame:
    """Fetch OHLCV from Polygon's aggregates endpoint.

    Crypto symbols use the 'X:' prefix (e.g., 'X:BTCUSD'); equities are bare
    tickers. We pass the symbol through as-is — caller's responsibility.
    """
    api_key = config.get("POLYGON_API_KEY", required=True)
    multiplier, span = TIMEFRAME_MAP.get(timeframe, (1, "day"))

    url = f"{POLYGON_API}/v2/aggs/ticker/{symbol.upper()}/range/{multiplier}/{span}/{start}/{end}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }

    with httpx.Client(timeout=30.0) as client:
        resp = client.get(url, params=params)
        # Don't leak the API key into logs/exceptions on failure
        if resp.status_code == 401:
            raise RuntimeError("Polygon: 401 Unauthorized. Check POLYGON_API_KEY in .env.")
        if resp.status_code == 429:
            raise RuntimeError("Polygon: 429 rate-limited. Free tier is 5 req/min.")
        resp.raise_for_status()
        body = resp.json()

    results = body.get("results") or []
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].astype(float)
    df.index.name = "date"
    return df
