"""Polymarket Gamma API adapter — public market data, no auth."""
import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"


def _normalize_category(raw: str | None) -> str:
    if not raw:
        return "Other"
    raw = raw.strip()
    mapping = {
        "Politics": "Politics",
        "Crypto": "Crypto",
        "Cryptocurrency": "Crypto",
        "Sports": "Sports",
        "Tech": "Tech",
        "Technology": "Tech",
        "Economics": "Economics",
        "Business": "Economics",
        "Finance": "Economics",
        "Geopolitics": "Geopolitics",
        "World": "Geopolitics",
        "Climate": "Climate",
        "Pop Culture": "Culture",
        "Culture": "Culture",
    }
    return mapping.get(raw, raw)


async def fetch_markets(limit: int = 100) -> list[dict[str, Any]]:
    """Fetch active, open Polymarket markets ordered by 24h volume."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{GAMMA_API}/markets",
            params={
                "active": "true",
                "closed": "false",
                "limit": limit,
                "order": "volume24hr",
                "ascending": "false",
            },
        )
        resp.raise_for_status()
        raw = resp.json()

    out: list[dict[str, Any]] = []
    for m in raw:
        try:
            prices_raw = m.get("outcomePrices") or "[]"
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            yes_price = float(prices[0]) if prices else 0.5

            end_iso = m.get("endDate")
            days_to_resolution: int | None = None
            if end_iso:
                try:
                    end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
                    days_to_resolution = max(0, (end_dt - datetime.now(timezone.utc)).days)
                except ValueError:
                    pass

            spread_raw = m.get("spread")
            spread = float(spread_raw) * 100 if spread_raw is not None else 0.0

            one_day_change = m.get("oneDayPriceChange")
            price_move_24h = float(one_day_change) * 100 if one_day_change is not None else 0.0

            out.append({
                "id": str(m.get("id") or m.get("conditionId") or ""),
                "question": m.get("question") or "",
                "category": _normalize_category(m.get("category")),
                "source": "Polymarket",
                "marketPrice": round(yes_price * 100, 2),
                "spread": round(spread, 2),
                "priceMove24h": round(price_move_24h, 2),
                "volume": float(m.get("volume24hr") or 0),
                "totalVolume": float(m.get("volumeNum") or 0),
                "liquidity": float(m.get("liquidity") or 0),
                "daysToResolution": days_to_resolution,
                "endDate": end_iso,
                "slug": m.get("slug"),
            })
        except Exception as e:
            logger.warning("Skipping Polymarket market %s: %s", m.get("id"), e)
            continue

    return out
