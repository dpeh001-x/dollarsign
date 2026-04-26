"""Reddit adapter — OAuth client_credentials flow with VADER sentiment scoring.

Returns recent posts from finance/politics/geopolitics subs with a compound
sentiment score in [-1, 1]. If REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are
not set, returns an empty feed (caller should fall back).
"""
import logging
import os
import time
from typing import Any

import httpx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

REDDIT_OAUTH_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_API = "https://oauth.reddit.com"

DEFAULT_SUBREDDITS = [
    "wallstreetbets",
    "options",
    "stocks",
    "investing",
    "geopolitics",
    "politics",
    "CryptoCurrency",
    "PredictionMarkets",
]

_analyzer = SentimentIntensityAnalyzer()
_token_cache: dict[str, Any] = {"token": None, "expires_at": 0}


async def _get_token(client: httpx.AsyncClient) -> str | None:
    now = time.time()
    if _token_cache["token"] and _token_cache["expires_at"] > now + 60:
        return _token_cache["token"]

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None

    user_agent = os.getenv("REDDIT_USER_AGENT", "oracle-prediction-platform/0.1")
    resp = await client.post(
        REDDIT_OAUTH_URL,
        auth=(client_id, client_secret),
        data={"grant_type": "client_credentials"},
        headers={"User-Agent": user_agent},
    )
    resp.raise_for_status()
    body = resp.json()
    token = body.get("access_token")
    expires_in = body.get("expires_in", 3600)
    _token_cache["token"] = token
    _token_cache["expires_at"] = now + expires_in
    return token


async def fetch_recent_posts(
    query: str | None = None,
    subreddits: list[str] | None = None,
    limit: int = 30,
) -> list[dict[str, Any]]:
    user_agent = os.getenv("REDDIT_USER_AGENT", "oracle-prediction-platform/0.1")
    subs = subreddits or DEFAULT_SUBREDDITS
    per_sub = max(2, limit // len(subs))

    out: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": user_agent}) as client:
        try:
            token = await _get_token(client)
        except httpx.HTTPError as e:
            logger.error("Reddit OAuth failed: %s", e)
            return []

        if not token:
            logger.warning("Reddit credentials not configured (REDDIT_CLIENT_ID/SECRET); empty feed")
            return []

        client.headers["Authorization"] = f"Bearer {token}"

        for sub in subs:
            try:
                if query:
                    url = f"{REDDIT_API}/r/{sub}/search"
                    params: dict[str, Any] = {
                        "q": query,
                        "restrict_sr": "1",
                        "sort": "new",
                        "t": "week",
                        "limit": per_sub,
                    }
                else:
                    url = f"{REDDIT_API}/r/{sub}/hot"
                    params = {"limit": per_sub}

                resp = await client.get(url, params=params)
                if resp.status_code == 429:
                    logger.warning("Reddit rate-limited on r/%s", sub)
                    continue
                if resp.status_code != 200:
                    logger.warning("Reddit r/%s returned %d", sub, resp.status_code)
                    continue
                data = resp.json()
            except httpx.HTTPError as e:
                logger.warning("Reddit r/%s error: %s", sub, e)
                continue

            for child in data.get("data", {}).get("children", []):
                post = child.get("data", {})
                title = (post.get("title") or "").strip()
                selftext = (post.get("selftext") or "").strip()
                if not title:
                    continue

                text_for_sent = f"{title} {selftext}"[:512]
                sent = _analyzer.polarity_scores(text_for_sent)

                out.append({
                    "src": "Reddit",
                    "user": f"r/{sub}",
                    "text": title,
                    "url": f"https://reddit.com{post.get('permalink', '')}",
                    "sent": round(sent["compound"], 2),
                    "score": int(post.get("score") or 0),
                    "comments": int(post.get("num_comments") or 0),
                    "createdUtc": float(post.get("created_utc") or 0),
                })

    out.sort(key=lambda x: x["createdUtc"], reverse=True)
    return out[:limit]
