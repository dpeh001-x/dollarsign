"""Oracle prediction platform — FastAPI backend.

Run locally:
    pip install -r requirements.txt
    cp .env.example .env  # then fill in Reddit credentials
    uvicorn main:app --reload --port 8000
"""
import asyncio
import logging
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from adapters import kalshi, polymarket, reddit  # noqa: E402

app = FastAPI(title="Oracle Prediction Platform Backend", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "service": "oracle-backend"}


@app.get("/api/markets")
async def get_markets(
    limit: int = Query(100, ge=1, le=500),
    sources: str = Query("polymarket,kalshi", description="comma-separated"),
) -> dict[str, Any]:
    """Fetch live markets from selected sources, run in parallel."""
    requested = {s.strip().lower() for s in sources.split(",") if s.strip()}

    tasks: dict[str, asyncio.Task] = {}
    if "polymarket" in requested:
        tasks["polymarket"] = asyncio.create_task(polymarket.fetch_markets(limit=limit))
    if "kalshi" in requested:
        tasks["kalshi"] = asyncio.create_task(kalshi.fetch_markets(limit=limit))

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    markets: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    errors: dict[str, str] = {}
    for source, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            errors[source] = f"{type(result).__name__}: {result}"
            counts[source] = 0
        else:
            markets.extend(result)
            counts[source] = len(result)

    counts["total"] = len(markets)
    return {"markets": markets, "counts": counts, "errors": errors}


@app.get("/api/sentiment")
async def get_sentiment(
    query: str | None = Query(None, description="optional keyword to search subs for"),
    limit: int = Query(30, ge=1, le=100),
) -> dict[str, Any]:
    """Recent Reddit posts with VADER sentiment scoring + aggregate."""
    posts = await reddit.fetch_recent_posts(query=query, limit=limit)

    if posts:
        agg = sum(p["sent"] for p in posts) / len(posts)
        positive = sum(1 for p in posts if p["sent"] > 0.3)
        negative = sum(1 for p in posts if p["sent"] < -0.3)
    else:
        agg = 0.0
        positive = 0
        negative = 0

    return {
        "feed": posts,
        "aggregate": round(agg, 3),
        "positive": positive,
        "negative": negative,
        "neutral": len(posts) - positive - negative,
        "count": len(posts),
        "source": "reddit",
        "configured": bool(posts) or _reddit_configured(),
    }


def _reddit_configured() -> bool:
    import os
    return bool(os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET"))


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Quick check on each upstream API. Slow — useful for diagnostics, not polling."""
    poly_ok = False
    kalshi_ok = False
    reddit_ok = _reddit_configured()
    poly_count = 0
    kalshi_count = 0

    try:
        poly_markets = await polymarket.fetch_markets(limit=5)
        poly_ok = True
        poly_count = len(poly_markets)
    except Exception as e:
        logger.warning("Polymarket health check failed: %s", e)

    try:
        kalshi_markets = await kalshi.fetch_markets(limit=5)
        kalshi_ok = True
        kalshi_count = len(kalshi_markets)
    except Exception as e:
        logger.warning("Kalshi health check failed: %s", e)

    return {
        "polymarket": {"ok": poly_ok, "sample_count": poly_count},
        "kalshi": {"ok": kalshi_ok, "sample_count": kalshi_count},
        "reddit": {"configured": reddit_ok},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
