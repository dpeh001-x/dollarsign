"""Kalshi adapter — public market discovery via the elections API.

Read-only market metadata (ticker, title, prices, volume, close_time) is
exposed without authentication. We fetch via the /events endpoint with
nested markets so each market arrives pre-categorized; this also avoids
the firehose of zero-volume sports player-prop tickers in /markets.
Trading and private user data require an API key + RSA-signed requests
(out of scope here).
"""
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

CATEGORY_MAP = {
    "Elections": "Politics",
    "Politics": "Politics",
    "Economics": "Economics",
    "Financials": "Economics",
    "Crypto": "Crypto",
    "Climate and Weather": "Climate",
    "Climate": "Climate",
    "Science and Technology": "Tech",
    "World": "Geopolitics",
    "Sports": "Sports",
    "Entertainment": "Culture",
    "Companies": "Tech",
    "Health": "Other",
}


def _category_from_ticker(ticker: str) -> str:
    if not ticker:
        return "Other"
    prefix = ticker.split("-", 1)[0].upper()
    mapping = {
        "PRES": "Politics",
        "GOV": "Politics",
        "SENATE": "Politics",
        "HOUSE": "Politics",
        "TRUMP": "Politics",
        "BIDEN": "Politics",
        "FED": "Economics",
        "CPI": "Economics",
        "GDP": "Economics",
        "JOBS": "Economics",
        "BTC": "Crypto",
        "ETH": "Crypto",
        "OIL": "Economics",
        "GAS": "Economics",
        "TEMP": "Climate",
        "NFL": "Sports",
        "NBA": "Sports",
        "MLB": "Sports",
    }
    return mapping.get(prefix, "Other")


async def fetch_markets(limit: int = 100, min_total_volume: float = 1000.0) -> list[dict[str, Any]]:
    """Fetch Kalshi markets via /events?with_nested_markets=true.

    Filters to non-trivial total volume (>=$1000 default), since 24h volume
    is sparse on prediction markets (action concentrates near resolution).
    Sorts by total volume descending.
    """
    collected: list[dict[str, Any]] = []
    cursor: str | None = None
    page_size = 200
    pages_seen = 0
    max_pages = 10

    async with httpx.AsyncClient(timeout=30.0) as client:
        while pages_seen < max_pages:
            params: dict[str, Any] = {
                "status": "open",
                "limit": page_size,
                "with_nested_markets": "true",
            }
            if cursor:
                params["cursor"] = cursor

            try:
                resp = await client.get(f"{KALSHI_API}/events", params=params)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as e:
                logger.error("Kalshi returned %d: %s", e.response.status_code, e.response.text[:200])
                break
            except httpx.HTTPError as e:
                logger.error("Kalshi network error: %s", e)
                break

            events = data.get("events", [])
            if not events:
                break

            for ev in events:
                ev_category = CATEGORY_MAP.get(ev.get("category") or "", ev.get("category") or "Other")
                for m in ev.get("markets", []):
                    try:
                        normalized = _normalize_market(m, category=ev_category)
                        if normalized["totalVolume"] >= min_total_volume:
                            collected.append(normalized)
                    except Exception as e:
                        logger.warning("Skipping Kalshi market %s: %s", m.get("ticker"), e)

            pages_seen += 1
            cursor = data.get("cursor")
            if not cursor:
                break

    collected.sort(key=lambda m: m["totalVolume"], reverse=True)
    return collected[:limit]


def _to_float(v: Any, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _normalize_market(m: dict[str, Any], category: str | None = None) -> dict[str, Any]:
    # Kalshi exposes prices as strings in dollars (e.g., "0.5500" = 55¢).
    # Convert to cents for consistency with Polymarket.
    yes_bid_d = _to_float(m.get("yes_bid_dollars"))
    yes_ask_d = _to_float(m.get("yes_ask_dollars"))
    last_d = _to_float(m.get("last_price_dollars"))
    prev_d = _to_float(m.get("previous_price_dollars"))

    if yes_bid_d and yes_ask_d:
        mid_price = (yes_bid_d + yes_ask_d) * 50  # avg dollars * 100
        spread = abs(yes_ask_d - yes_bid_d) * 100
    elif last_d:
        mid_price = last_d * 100
        spread = 0.0
    else:
        mid_price = 50.0
        spread = 0.0

    price_move_24h = (last_d - prev_d) * 100 if (last_d and prev_d) else 0.0

    close_time = m.get("close_time")
    days_to_resolution: int | None = None
    if close_time:
        try:
            close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            days_to_resolution = max(0, (close_dt - datetime.now(timezone.utc)).days)
        except ValueError:
            pass

    ticker = m.get("ticker") or ""
    return {
        "id": ticker,
        "question": m.get("title") or "",
        "subtitle": m.get("subtitle"),
        "category": category or m.get("category") or _category_from_ticker(ticker),
        "source": "Kalshi",
        "marketPrice": round(mid_price, 2),
        "spread": round(spread, 2),
        "priceMove24h": round(price_move_24h, 2),
        "volume": _to_float(m.get("volume_24h_fp")),
        "totalVolume": _to_float(m.get("volume_fp")),
        "liquidity": _to_float(m.get("liquidity_dollars")),
        "openInterest": _to_float(m.get("open_interest_fp")),
        "daysToResolution": days_to_resolution,
        "endDate": close_time,
        "ticker": ticker,
    }
