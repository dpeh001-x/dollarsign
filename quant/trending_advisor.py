"""Score apewisdom trending tickers + emit an objective buy/avoid verdict.

Pure rule-based — no ML. For each trending ticker we pull cached daily OHLC,
compute trend + momentum scores via the trend_momentum analyzer, then combine
with the social mention dynamics to produce:

  verdict ∈ {BUY, ACCUMULATE, WATCH, AVOID, SHORT}
  thesis  — a 1–3 sentence objective summary citing the actual numbers.

Verdict logic (intentionally conservative):
  BUY        — trend aligned up, momentum bullish, social mom > 1, NOT extremely overbought
  ACCUMULATE — trend up but momentum cooling, OR overbought (wait for pullback)
  WATCH      — mixed technicals or insufficient data; social hype isn't enough alone
  AVOID      — trend down or weak with bearish momentum; the crowd is chasing a falling knife
  SHORT      — trend aligned down, momentum bearish, ADX confirms (rare; high bar)

Thesis lines are formatted as plain factual statements — no hype words.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import trend_momentum

logger = logging.getLogger(__name__)


@dataclass
class AdvisedTicker:
    ticker: str
    company_name: str
    # social
    mentions: int
    mentions_24h_ago: int
    social_momentum: float       # mentions / mentions_24h_ago
    social_score: float          # apewisdom composite
    # technical (None if we couldn't fetch data)
    last_close: float | None
    trend_score: float | None    # -1..+1
    trend_label: str | None
    momentum_score: float | None # -1..+1
    momentum_label: str | None
    rsi_14: float | None
    adx: float | None
    ret_5d: float | None
    ret_20d: float | None
    # verdict
    verdict: str                 # BUY | ACCUMULATE | WATCH | AVOID | SHORT
    verdict_color: str           # hex for UI
    conviction: int              # 1..5
    thesis: str                  # 1–3 sentence objective summary


# Skip well-known apewisdom junk that isn't a real listed ticker.
_BLOCKLIST = {"INTEL", "CEO", "USA", "AI", "USD", "ATH", "FED", "GDP",
              "WSB", "IPO", "ETF", "OK", "PR", "PE"}


def _verdict_and_thesis(
    ticker: str,
    company: str,
    mentions: int,
    mentions_24h: int,
    social_mom: float,
    tech: trend_momentum.TrendMomentumResult | None,
) -> tuple[str, str, int, str]:
    """Return (verdict, color, conviction, thesis)."""

    # No technical data — only social. Keep skeptical.
    if tech is None:
        thesis = (
            f"Trending in social ({mentions_24h}→{mentions} mentions, "
            f"{social_mom:.1f}× momentum) but we could not pull price data — "
            f"may be a non-equity ticker (crypto without -USD), OTC, or delisted. "
            f"Cannot form a technical view."
        )
        return "WATCH", "#FF9F43", 1, thesis

    t = tech.trend_score
    m = tech.momentum_score
    rsi = tech.rsi_14
    adx = tech.adx
    ret20 = tech.ret_20d * 100
    ret5 = tech.ret_5d * 100
    aligned_up = t > 0.25 and m > 0.20
    aligned_dn = t < -0.25 and m < -0.20
    overbought = rsi >= 75
    oversold = rsi <= 25

    # Build a factual prefix shared by all verdicts.
    social_clause = (
        f"Social: {mentions_24h}→{mentions} mentions ({social_mom:.1f}× momentum)."
    )
    tech_clause = (
        f"Tech: trend {tech.trend_label.replace('_',' ')} ({t:+.2f}), "
        f"momentum {tech.momentum_label.replace('_',' ')} ({m:+.2f}); "
        f"RSI {rsi:.0f}, ADX {adx:.0f}, 5d {ret5:+.1f}%, 20d {ret20:+.1f}%."
    )

    # ─── Decision tree ───
    if aligned_up and not overbought and social_mom > 1.0:
        verdict = "BUY"
        color = "#00E68A"
        conv = 5 if (t > 0.5 and m > 0.4 and adx > 25) else 4
        action = (
            f"Trend & momentum both lean up with ADX {adx:.0f} confirming, social interest is heating up. "
            f"Reasonable long candidate; size with 2.5×ATR stop. Risk: 20d return of {ret20:+.1f}% means entry isn't early."
        )
        return verdict, color, conv, f"{social_clause} {tech_clause} {action}"

    if aligned_up and overbought:
        verdict = "ACCUMULATE"
        color = "#4FACFE"
        conv = 3
        action = (
            f"Trend is up but RSI {rsi:.0f} is overbought after a {ret20:+.1f}% 20-day run. "
            f"Crowd is late — wait for a pullback toward SMA50 (${tech.sma50:.2f}) before committing."
        )
        return verdict, color, conv, f"{social_clause} {tech_clause} {action}"

    if aligned_up and social_mom <= 1.0:
        verdict = "ACCUMULATE"
        color = "#4FACFE"
        conv = 3
        action = (
            f"Technicals are constructive but social interest is fading ({social_mom:.1f}× momentum). "
            f"Real setup, but the social-trade angle is weakening — buy on fundamentals, not on the crowd."
        )
        return verdict, color, conv, f"{social_clause} {tech_clause} {action}"

    if aligned_dn and adx > 20:
        verdict = "SHORT"
        color = "#FF4757"
        conv = 4 if adx > 25 else 3
        action = (
            f"Trend & momentum both lean down with ADX {adx:.0f} confirming a real downtrend. "
            f"Crowd is chasing a falling knife; short candidate with tight stop above SMA50 (${tech.sma50:.2f})."
        )
        return verdict, color, conv, f"{social_clause} {tech_clause} {action}"

    if aligned_dn:
        verdict = "AVOID"
        color = "#FF4757"
        conv = 3
        action = (
            f"Trend & momentum agree to the downside but ADX {adx:.0f} is weak — could chop sideways. "
            f"No long here; not worth shorting without trend confirmation."
        )
        return verdict, color, conv, f"{social_clause} {tech_clause} {action}"

    if oversold and t > -0.2:
        verdict = "WATCH"
        color = "#FFD93D"
        conv = 2
        action = (
            f"RSI {rsi:.0f} is oversold but trend isn't yet bullish ({t:+.2f}). "
            f"Possible bounce setup — wait for trend confirmation (close > SMA50 ${tech.sma50:.2f}) before entering."
        )
        return verdict, color, conv, f"{social_clause} {tech_clause} {action}"

    if t < -0.25 and m > 0:
        verdict = "WATCH"
        color = "#FFD93D"
        conv = 2
        action = (
            f"Downtrend but short-term momentum is positive ({m:+.2f}) — could be a dead-cat bounce or genuine reversal. "
            f"Don't front-run; wait for SMA50 reclaim."
        )
        return verdict, color, conv, f"{social_clause} {tech_clause} {action}"

    if t > 0.25 and m < 0:
        verdict = "WATCH"
        color = "#FFD93D"
        conv = 2
        action = (
            f"Uptrend intact but short-term momentum has rolled over ({m:+.2f}). "
            f"Pause — either a healthy pullback or distribution. Re-evaluate when momentum turns."
        )
        return verdict, color, conv, f"{social_clause} {tech_clause} {action}"

    # Catch-all: ranging / no edge
    verdict = "WATCH"
    color = "#FF9F43"
    conv = 1
    action = (
        f"No clean directional edge — trend {t:+.2f}, momentum {m:+.2f}. "
        f"Social attention without technical setup is just noise."
    )
    return verdict, color, conv, f"{social_clause} {tech_clause} {action}"


def _resolve_for_data(ticker: str) -> str | None:
    """Map a social-feed ticker to a fetchable symbol, or None to skip."""
    t = ticker.upper().strip()
    if not t or t in _BLOCKLIST:
        return None
    # Already in yfinance crypto form (e.g. BTC-USD)? Use as-is.
    if "-USD" in t:
        return t
    # Skip obvious non-tickers (numbers, single chars except a couple legit ones)
    if not t.replace(".", "").replace("-", "").isalpha():
        return None
    if len(t) > 6:
        return None
    return t


def advise(trending: list, max_advice: int = 15) -> list[AdvisedTicker]:
    """Enrich apewisdom TickerSignal results with technical analysis + verdict.

    Args:
        trending: list of social_scanner.TickerSignal (or dict-like with same fields)
        max_advice: cap on how many to analyze (technical analysis is the slow step)

    Returns:
        List of AdvisedTicker, in the same order as `trending` (truncated to max_advice).
    """
    start_iso = (date.today() - timedelta(days=3 * 365)).isoformat()
    out: list[AdvisedTicker] = []

    for sig in trending[:max_advice]:
        # Tolerate either TickerSignal dataclass or dict
        ticker = getattr(sig, "ticker", None) or sig["ticker"]
        company = getattr(sig, "company_name", None) or sig.get("company", "")
        mentions = getattr(sig, "mentions", None) if hasattr(sig, "mentions") else sig["mentions"]
        m24 = (getattr(sig, "mentions_24h_ago", None)
               if hasattr(sig, "mentions_24h_ago") else sig.get("yesterday", 0))
        social_mom = getattr(sig, "momentum", None) if hasattr(sig, "momentum") else sig.get("momentum", 1.0)
        social_score = getattr(sig, "score", None) if hasattr(sig, "score") else sig.get("score", 0.0)

        fetchable = _resolve_for_data(ticker)
        tech: trend_momentum.TrendMomentumResult | None = None
        if fetchable is not None:
            try:
                tech = trend_momentum._analyze_symbol(fetchable, "trending", start_iso)
            except Exception as e:
                logger.warning("trend_momentum %s: %s", ticker, e)
                tech = None

        verdict, color, conv, thesis = _verdict_and_thesis(
            ticker, company, mentions, m24, social_mom, tech,
        )

        out.append(AdvisedTicker(
            ticker=ticker, company_name=company,
            mentions=mentions, mentions_24h_ago=m24,
            social_momentum=social_mom, social_score=social_score,
            last_close=(tech.last_close if tech else None),
            trend_score=(tech.trend_score if tech else None),
            trend_label=(tech.trend_label if tech else None),
            momentum_score=(tech.momentum_score if tech else None),
            momentum_label=(tech.momentum_label if tech else None),
            rsi_14=(tech.rsi_14 if tech else None),
            adx=(tech.adx if tech else None),
            ret_5d=(tech.ret_5d if tech else None),
            ret_20d=(tech.ret_20d if tech else None),
            verdict=verdict, verdict_color=color, conviction=conv,
            thesis=thesis,
        ))

    return out


if __name__ == "__main__":
    import social_scanner
    sigs = social_scanner.scan(limit=10, filter_name="all-stocks")
    print(f"\nAdvising on top {len(sigs)} trending tickers...\n")
    advised = advise(sigs)
    for a in advised:
        print(f"\n{'='*70}")
        print(f"  {a.ticker:<8} {a.company_name}")
        print(f"  VERDICT: {a.verdict}  (conviction {a.conviction}/5)")
        print(f"  {a.thesis}")
