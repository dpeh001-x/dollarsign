"""OHLCV data sources: Alpaca + Polygon + yfinance, parquet-cached.

API keys are loaded via `config.get()` so they're never read from source
files or hardcoded. See quant/config.py for the loader.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from data import polygon as polygon_src  # noqa: E402

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(symbol: str, start: str, end: str, timeframe: str) -> Path:
    safe = symbol.upper().replace("/", "_")
    return CACHE_DIR / f"{safe}_{timeframe}_{start}_{end}.parquet"


def _fetch_alpaca(symbol: str, start: str, end: str, timeframe: str = "1Day") -> pd.DataFrame:
    """Pull bars from Alpaca's market data v2 API."""
    config.assert_configured("alpaca")
    key_id = config.get("ALPACA_API_KEY_ID", required=True)
    secret = config.get("ALPACA_API_SECRET_KEY", required=True)
    base_url = config.get("ALPACA_DATA_URL", default="https://data.alpaca.markets/v2")

    url = f"{base_url}/stocks/{symbol.upper()}/bars"
    headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}
    params: dict[str, str | int] = {
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "limit": 10000,
        "adjustment": "all",
        "feed": "iex",  # free tier
    }

    all_bars: list[dict] = []
    next_token: str | None = None

    with httpx.Client(timeout=30.0, headers=headers) as client:
        while True:
            if next_token:
                params["page_token"] = next_token
            resp = client.get(url, params=params)
            if resp.status_code == 401:
                raise RuntimeError("Alpaca: 401 Unauthorized. Check ALPACA_API_KEY_ID/SECRET in .env.")
            resp.raise_for_status()
            payload = resp.json()
            all_bars.extend(payload.get("bars") or [])
            next_token = payload.get("next_page_token")
            if not next_token:
                break

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars)
    df["timestamp"] = pd.to_datetime(df["t"], utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].astype(float)
    df.index.name = "date"
    return df


def _fetch_yfinance(symbol: str, start: str, end: str, timeframe: str = "1Day") -> pd.DataFrame:
    """Fallback: yfinance daily/hourly bars."""
    import yfinance as yf

    interval_map = {"1Day": "1d", "1Hour": "1h", "1Min": "1m"}
    interval = interval_map.get(timeframe, "1d")

    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]].astype(float)


SOURCE_ORDER_AUTO = ("alpaca", "polygon", "yfinance")


def get_bars(
    symbol: str,
    start: str,
    end: str | None = None,
    timeframe: str = "1Day",
    use_cache: bool = True,
    prefer: str = "auto",  # "auto" | "alpaca" | "polygon" | "yfinance"
) -> pd.DataFrame:
    """Get OHLCV bars. Tries Alpaca -> Polygon -> yfinance in auto mode.

    `prefer` values other than "auto" force a specific source (raises if its
    keys aren't configured). `start`/`end` are YYYY-MM-DD strings.
    """
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    cache_file = _cache_path(symbol, start, end, timeframe)
    if use_cache and cache_file.exists():
        logger.info("Cache hit: %s", cache_file.name)
        return pd.read_parquet(cache_file)

    if prefer == "auto":
        candidates = SOURCE_ORDER_AUTO
    else:
        candidates = (prefer,)

    df = pd.DataFrame()
    source_used = None
    last_error: Exception | None = None

    for src in candidates:
        try:
            if src == "alpaca":
                if not config.PROVIDERS["alpaca"].is_configured():
                    if prefer == "auto":
                        continue
                    raise RuntimeError("Alpaca not configured")
                df = _fetch_alpaca(symbol, start, end, timeframe)
            elif src == "polygon":
                if not config.PROVIDERS["polygon"].is_configured():
                    if prefer == "auto":
                        continue
                    raise RuntimeError("Polygon not configured")
                df = polygon_src.fetch_bars(symbol, start, end, timeframe)
            elif src == "yfinance":
                df = _fetch_yfinance(symbol, start, end, timeframe)
            else:
                raise ValueError(f"Unknown source: {src}")

            if not df.empty:
                source_used = src
                break
        except Exception as e:
            last_error = e
            logger.warning("%s fetch failed: %s", src, e)
            if prefer != "auto":
                raise

    if df.empty:
        raise RuntimeError(f"No data returned for {symbol} from any source. Last error: {last_error}")

    df.attrs["source"] = source_used
    df.attrs["symbol"] = symbol.upper()

    if use_cache:
        df.to_parquet(cache_file)

    logger.info("Loaded %d bars for %s from %s", len(df), symbol, source_used)
    return df
