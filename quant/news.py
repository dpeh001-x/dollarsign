"""NewsAPI.org adapter with VADER sentiment scoring.

Free dev tier: 100 requests/day, 24h article delay, dev environments only.
Don't use it in production — Polygon also has a /reference/news endpoint
that's better for stock-tagged news; consider it a future addition.

The API key is loaded via config.get(), never read directly from os.getenv()
or hardcoded. Empty/missing key raises a clear error rather than silently
returning fake data.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config

logger = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"
_analyzer = SentimentIntensityAnalyzer()


def fetch_articles(
    query: str,
    days_back: int = 7,
    page_size: int = 50,
    language: str = "en",
    sort_by: str = "publishedAt",
) -> list[dict[str, Any]]:
    """Search NewsAPI for articles matching `query`. Returns scored results.

    Each item: {title, description, source, url, publishedAt, sent_compound,
    sent_label}. sent_compound is in [-1, 1]; sent_label is bullish/bearish/neutral.
    """
    api_key = config.get("NEWSAPI_KEY", required=True)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)

    params = {
        "q": query,
        "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "language": language,
        "sortBy": sort_by,
        "pageSize": min(page_size, 100),
    }
    headers = {"X-Api-Key": api_key}  # header auth keeps the key out of URL logs

    with httpx.Client(timeout=30.0, headers=headers) as client:
        resp = client.get(NEWSAPI_URL, params=params)
        if resp.status_code == 401:
            raise RuntimeError("NewsAPI: 401 Unauthorized. Check NEWSAPI_KEY in .env.")
        if resp.status_code == 426:
            raise RuntimeError("NewsAPI: 426 Upgrade Required. Free tier may not support this query/range.")
        if resp.status_code == 429:
            raise RuntimeError("NewsAPI: 429 rate-limited. Free tier is 100 req/day.")
        resp.raise_for_status()
        body = resp.json()

    if body.get("status") != "ok":
        raise RuntimeError(f"NewsAPI error: {body.get('message', body)}")

    out: list[dict[str, Any]] = []
    for art in body.get("articles", []):
        title = (art.get("title") or "").strip()
        desc = (art.get("description") or "").strip()
        if not title:
            continue

        text = f"{title}. {desc}"[:1024]
        scores = _analyzer.polarity_scores(text)
        compound = scores["compound"]
        label = "bullish" if compound > 0.3 else "bearish" if compound < -0.3 else "neutral"

        out.append({
            "title": title,
            "description": desc,
            "source": (art.get("source") or {}).get("name") or "unknown",
            "url": art.get("url"),
            "publishedAt": art.get("publishedAt"),
            "sent_compound": round(compound, 3),
            "sent_label": label,
        })

    return out


def aggregate_sentiment(articles: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate sentiment over a list of articles."""
    if not articles:
        return {"count": 0, "mean_compound": 0.0, "bullish": 0, "bearish": 0, "neutral": 0}

    n = len(articles)
    mean = sum(a["sent_compound"] for a in articles) / n
    bullish = sum(1 for a in articles if a["sent_label"] == "bullish")
    bearish = sum(1 for a in articles if a["sent_label"] == "bearish")
    neutral = n - bullish - bearish

    return {
        "count": n,
        "mean_compound": round(mean, 3),
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
        "label": "bullish" if mean > 0.15 else "bearish" if mean < -0.15 else "neutral",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="search query, e.g. 'AAPL' or 'federal reserve'")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    arts = fetch_articles(args.query, days_back=args.days, page_size=args.limit)
    agg = aggregate_sentiment(arts)
    print(f"\nQuery: {args.query}  ({args.days}d)")
    print(f"Aggregate: {agg['label']:>8s}  mean={agg['mean_compound']:+.3f}  "
          f"({agg['bullish']} bull / {agg['bearish']} bear / {agg['neutral']} neut)\n")
    for a in arts[:args.limit]:
        sign = "+" if a["sent_compound"] > 0 else ""
        print(f"  [{sign}{a['sent_compound']:>5.2f}] {a['source'][:18]:<18s} {a['title'][:80]}")
