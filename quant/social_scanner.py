"""apewisdom.io ticker scanner — trending stocks across Reddit, StockTwits, Twitter.

apewisdom aggregates ticker mentions across all major finance social sources
into a single ranked feed. No API key needed, no Cloudflare block. Each
result carries today's mentions AND mentions-24h-ago, so we can rank by
*momentum* (heating up) rather than just raw popularity.

Filters available:
    all-stocks (default), wallstreetbets, stocks, options, smallstreetbets,
    securityanalysis, valueinvesting, daytrading, cryptos

Composite score: mentions weighted by 24h momentum.
    momentum = mentions / max(mentions_24h_ago, 1)
    score    = sqrt(mentions) * (0.5 + 0.5 * tanh(log(momentum)))

The tanh(log) shape gives a smooth boost for momentum > 1 (heating up),
penalty for momentum < 1 (cooling off), and zero at momentum = 1 (flat).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

APEWISDOM_API = "https://apewisdom.io/api/v1.0"
USER_AGENT = "dollarsign-scanner/0.1"

# Crypto symbol mapping for the ML pipeline (yfinance convention)
CRYPTO_REMAP = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD",
    "DOGE": "DOGE-USD", "ADA": "ADA-USD", "AVAX": "AVAX-USD",
    "XRP": "XRP-USD", "LTC": "LTC-USD", "DOT": "DOT-USD",
    "MATIC": "MATIC-USD", "SHIB": "SHIB-USD",
}

VALID_FILTERS = (
    "all-stocks", "wallstreetbets", "stocks", "options",
    "smallstreetbets", "securityanalysis", "valueinvesting",
    "daytrading", "cryptos",
)


def _is_configured() -> bool:
    """No auth needed — always True."""
    return True


@dataclass
class TickerSignal:
    ticker: str
    company_name: str
    mentions: int
    mentions_24h_ago: int
    upvotes: int
    momentum: float       # mentions / mentions_24h_ago (>1 = heating up)
    rank: int
    rank_24h_ago: int
    rank_change: int      # rank_24h_ago - rank (+ = climbed)
    score: float


def _normalize_ticker(raw: str, filter_name: str) -> str | None:
    raw = (raw or "").upper().strip()
    if not raw:
        return None
    # Map crypto to yfinance form for crypto filter
    if filter_name == "cryptos" and raw in CRYPTO_REMAP:
        return CRYPTO_REMAP[raw]
    return raw


def scan(limit: int = 25, filter_name: str = "all-stocks") -> list[TickerSignal]:
    """Fetch top-ranked tickers from apewisdom.

    Args:
        limit: max number of tickers to return.
        filter_name: which source to scan. See VALID_FILTERS for options.

    Returns:
        List of TickerSignal sorted by composite score (highest first).
    """
    if filter_name not in VALID_FILTERS:
        raise ValueError(f"filter_name must be one of {VALID_FILTERS}, got: {filter_name}")

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    url = f"{APEWISDOM_API}/filter/{filter_name}/page/1"

    try:
        with httpx.Client(timeout=30.0, headers=headers) as client:
            resp = client.get(url)
            if resp.status_code == 429:
                raise RuntimeError("apewisdom rate-limited (429). Retry in a minute.")
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        raise RuntimeError(f"apewisdom API unreachable: {e}") from e

    out: list[TickerSignal] = []
    for r in data.get("results", []):
        ticker = _normalize_ticker(r.get("ticker", ""), filter_name)
        if not ticker:
            continue

        mentions = int(r.get("mentions") or 0)
        mentions_24h = int(r.get("mentions_24h_ago") or 0)
        upvotes = int(r.get("upvotes") or 0)
        rank = int(r.get("rank") or 0)
        rank_24h = int(r.get("rank_24h_ago") or 0)

        momentum = mentions / max(mentions_24h, 1)
        log_mom = math.log(max(momentum, 0.01))
        score = math.sqrt(mentions) * (0.5 + 0.5 * math.tanh(log_mom))

        name = (r.get("name") or "").replace("&amp;", "&").strip()
        rank_change = (rank_24h - rank) if (rank_24h > 0 and rank > 0) else 0

        out.append(TickerSignal(
            ticker=ticker,
            company_name=name[:60],
            mentions=mentions,
            mentions_24h_ago=mentions_24h,
            upvotes=upvotes,
            momentum=round(momentum, 2),
            rank=rank,
            rank_24h_ago=rank_24h,
            rank_change=rank_change,
            score=round(score, 2),
        ))

    out.sort(key=lambda r: r.score, reverse=True)
    return out[:limit]


if __name__ == "__main__":
    import sys
    filter_arg = sys.argv[1] if len(sys.argv) > 1 else "all-stocks"
    print(f"\nScanning apewisdom (filter={filter_arg})...\n")
    try:
        results = scan(limit=20, filter_name=filter_arg)
    except (ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("(no tickers found)")
    else:
        print(f"{'#':>3} {'ticker':<10} {'co':<28} {'now':>5} {'24h':>5} {'mom':>5} {'rank+/-':>7} {'score':>6}")
        print(f"{'-'*3} {'-'*10} {'-'*28} {'-'*5} {'-'*5} {'-'*5} {'-'*7} {'-'*6}")
        for i, r in enumerate(results, 1):
            co = (r.company_name[:25] + "...") if len(r.company_name) > 28 else r.company_name
            rc = f"{r.rank_change:+d}" if r.rank_change != 0 else "-"
            print(f"{i:>3} {r.ticker:<10} {co:<28} {r.mentions:>5} {r.mentions_24h_ago:>5} {r.momentum:>5.1f} {rc:>7} {r.score:>6.2f}")
