"""Options trading utilities — Black-Scholes pricing, Greeks, IV solver,
live chain fetching (yfinance), and concrete strategy suggestions tied to
the model's directional signal.

Conventions:
    S      spot price (underlying)
    K      strike
    T      time to expiry, in YEARS (so 30 days = 30/365)
    r      risk-free rate (annualized, decimal: 0.045 = 4.5%)
    sigma  volatility (annualized, decimal: 0.25 = 25%)
    Premium values are PER SHARE (multiply by 100 for the contract value).

Educational tool. Real options trading involves the volatility smile, early
exercise (American style), dividends, transaction costs, assignment risk,
margin requirements — none of which are modeled here. Nothing here is
financial advice.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

logger = logging.getLogger(__name__)

DEFAULT_RISK_FREE_RATE = 0.045  # roughly current SOFR/3M T-bill yield


# ─── Black-Scholes pricing & Greeks ────────────────────────────────────
def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """European-style Black-Scholes price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> dict[str, float]:
    """Standard Greeks. theta is per CALENDAR DAY; vega per 1 percentage point of vol; rho per 1 pp of rate."""
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta / 365),  # per calendar day
        "vega": float(vega / 100),    # per 1 pp vol change
        "rho": float(rho / 100),      # per 1 pp rate change
    }


def implied_volatility(market_price: float, S: float, K: float, T: float, r: float, option_type: str = "call") -> float:
    """Solve for implied vol using Brent's method. Returns NaN on failure / arbitrage."""
    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if market_price < intrinsic - 1e-6 or T <= 0:
        return float("nan")
    try:
        f = lambda sig: bs_price(S, K, T, r, sig, option_type) - market_price
        return float(brentq(f, 1e-4, 5.0, xtol=1e-6, maxiter=100))
    except (ValueError, RuntimeError):
        return float("nan")


# ─── Live options chain (yfinance) ─────────────────────────────────────
def get_options_chain(symbol: str, n_expirations: int = 6) -> dict[str, dict[str, pd.DataFrame]]:
    """Returns {expiration_date_str: {'calls': df, 'puts': df}}.

    yfinance returns columns like: strike, lastPrice, bid, ask, change,
    volume, openInterest, impliedVolatility. Empty dict if symbol has no
    listed options (most non-equity symbols, including BTC-USD).
    """
    import yfinance as yf
    try:
        ticker = yf.Ticker(symbol)
        expirations = list(ticker.options) if ticker.options else []
    except Exception as e:
        logger.warning("options chain fetch failed for %s: %s", symbol, e)
        return {}

    chains: dict[str, dict[str, pd.DataFrame]] = {}
    for exp in expirations[:n_expirations]:
        try:
            opts = ticker.option_chain(exp)
            chains[exp] = {"calls": opts.calls, "puts": opts.puts}
        except Exception:
            continue
    return chains


def days_to_expiry(exp_str: str) -> int:
    """Calendar days until the given expiration (YYYY-MM-DD)."""
    exp = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return max(0, (exp - datetime.now(timezone.utc)).days)


def find_atm_iv(chain: dict[str, pd.DataFrame], spot: float) -> float | None:
    """ATM straddle implied vol — average of ATM call+put IV. Returns None if data missing."""
    calls, puts = chain.get("calls"), chain.get("puts")
    if calls is None or puts is None or "impliedVolatility" not in calls.columns:
        return None
    try:
        c_idx = (calls["strike"] - spot).abs().idxmin()
        p_idx = (puts["strike"] - spot).abs().idxmin()
        iv_c = float(calls.loc[c_idx, "impliedVolatility"]) or 0
        iv_p = float(puts.loc[p_idx, "impliedVolatility"]) or 0
        if iv_c > 0 and iv_p > 0:
            return (iv_c + iv_p) / 2
        return iv_c if iv_c > 0 else (iv_p if iv_p > 0 else None)
    except Exception:
        return None


def closest_strike(strikes: list[float], target: float) -> float:
    arr = np.array(strikes)
    return float(arr[np.abs(arr - target).argmin()])


def find_quote(chain_df: pd.DataFrame, strike: float, prefer_mid: bool = True) -> float | None:
    """Get a price quote for the given strike from the chain. Prefers midpoint of bid/ask, falls back to lastPrice."""
    matches = chain_df[chain_df["strike"] == strike]
    if matches.empty:
        return None
    row = matches.iloc[0]
    if prefer_mid and "bid" in row and "ask" in row:
        bid = float(row.get("bid") or 0)
        ask = float(row.get("ask") or 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
    last = float(row.get("lastPrice") or 0)
    return last if last > 0 else None


def expected_move(spot: float, vol_annual: float, days: int) -> float:
    """Approximate 1-std expected move in DOLLARS over `days`. vol_annual decimal."""
    return spot * vol_annual * np.sqrt(days / 365.0)


# ─── Strategy result type ──────────────────────────────────────────────
@dataclass
class Leg:
    side: str   # 'buy' or 'sell'
    type: str   # 'call' or 'put'
    strike: float
    premium: float  # per share

    def signed_premium(self) -> float:
        """+ for cash paid out (buy), - for cash received (sell)."""
        return self.premium if self.side == "buy" else -self.premium


@dataclass
class Strategy:
    name: str
    thesis: str           # 1-line plain-English
    legs: list[Leg]
    days_to_expiry: int
    net_debit: float      # + = pay, - = collect
    max_profit: float     # per share; inf if unlimited
    max_loss: float       # per share, positive number
    breakevens: list[float]
    iv_used: float | None = None       # decimal; what we used in any pricing

    def payoff(self, S: float) -> float:
        """P&L per share at expiry. Computed from legs (no captured closures
        so the dataclass is picklable for Streamlit's cache_data)."""
        total = 0.0
        for leg in self.legs:
            sign = 1.0 if leg.side == "buy" else -1.0
            intrinsic = _intrinsic(S, leg.strike, leg.type)
            total += sign * (intrinsic - leg.premium)
        return total


# ─── Strategy constructors ─────────────────────────────────────────────
def _intrinsic(S: float, K: float, opt: str) -> float:
    return max(S - K, 0.0) if opt == "call" else max(K - S, 0.0)


def long_call(spot: float, K: float, premium: float, days: int) -> Strategy:
    return Strategy(
        name="Long call",
        thesis=f"Bullish — pay ${premium:.2f}/share for the right to buy at ${K:.2f}. Profit if price > ${K + premium:.2f} at expiry.",
        legs=[Leg("buy", "call", K, premium)],
        days_to_expiry=days,
        net_debit=premium,
        max_profit=float("inf"),
        max_loss=premium,
        breakevens=[K + premium],
    )


def long_put(spot: float, K: float, premium: float, days: int) -> Strategy:
    return Strategy(
        name="Long put",
        thesis=f"Bearish — pay ${premium:.2f}/share for the right to sell at ${K:.2f}. Profit if price < ${K - premium:.2f} at expiry.",
        legs=[Leg("buy", "put", K, premium)],
        days_to_expiry=days,
        net_debit=premium,
        max_profit=K - premium,  # if S goes to 0
        max_loss=premium,
        breakevens=[K - premium],
    )


def bull_call_spread(spot: float, K_long: float, K_short: float, prem_long: float, prem_short: float, days: int) -> Strategy:
    assert K_short > K_long, "bull call spread: short strike must be above long strike"
    debit = prem_long - prem_short
    width = K_short - K_long
    return Strategy(
        name="Bull call spread",
        thesis=f"Bullish — buy ${K_long:.2f} call + sell ${K_short:.2f} call for net debit ${debit:.2f}. Cheaper than long call; capped reward.",
        legs=[Leg("buy", "call", K_long, prem_long), Leg("sell", "call", K_short, prem_short)],
        days_to_expiry=days,
        net_debit=debit,
        max_profit=width - debit,
        max_loss=debit,
        breakevens=[K_long + debit],
    )


def bear_put_spread(spot: float, K_long: float, K_short: float, prem_long: float, prem_short: float, days: int) -> Strategy:
    assert K_long > K_short, "bear put spread: long strike must be above short strike"
    debit = prem_long - prem_short
    width = K_long - K_short
    return Strategy(
        name="Bear put spread",
        thesis=f"Bearish — buy ${K_long:.2f} put + sell ${K_short:.2f} put for net debit ${debit:.2f}. Cheaper than long put; capped reward.",
        legs=[Leg("buy", "put", K_long, prem_long), Leg("sell", "put", K_short, prem_short)],
        days_to_expiry=days,
        net_debit=debit,
        max_profit=width - debit,
        max_loss=debit,
        breakevens=[K_long - debit],
    )


def iron_condor(spot: float, K_put_long: float, K_put_short: float, K_call_short: float, K_call_long: float,
                p_put_long: float, p_put_short: float, p_call_short: float, p_call_long: float, days: int) -> Strategy:
    """Sell put spread + sell call spread → collect premium, profit if price stays in middle."""
    credit = (p_put_short - p_put_long) + (p_call_short - p_call_long)
    width = max(K_put_short - K_put_long, K_call_long - K_call_short)
    max_loss = width - credit
    return Strategy(
        name="Iron condor",
        thesis=f"Neutral — collect ${credit:.2f} premium. Profit if price stays between ${K_put_short:.2f} and ${K_call_short:.2f}.",
        legs=[
            Leg("buy", "put", K_put_long, p_put_long),
            Leg("sell", "put", K_put_short, p_put_short),
            Leg("sell", "call", K_call_short, p_call_short),
            Leg("buy", "call", K_call_long, p_call_long),
        ],
        days_to_expiry=days,
        net_debit=-credit,  # negative because credit
        max_profit=credit,
        max_loss=max_loss,
        breakevens=[K_put_short - credit, K_call_short + credit],
    )


def spot_label(spot: float) -> str:
    return f"price"  # used in thesis text


# ─── Top-level: pick + build a concrete strategy from model output ─────
def suggest_strategy(
    symbol: str,
    spot: float,
    signal: str,
    confidence: float,
    horizon_days: int,
    historical_vol: float,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> dict:
    """Given the model's directional signal + confidence + expected move,
    suggest a concrete options play with real strikes & prices.

    Tries to use live yfinance options chain. Falls back to Black-Scholes
    theoretical pricing on hypothetical strikes if no chain available.

    Returns:
        {
            'strategy': Strategy object,
            'expiration': date string,
            'spot': float,
            'expected_move_pct': float,
            'iv_used': float | None,
            'iv_source': 'market' | 'historical' | 'none',
            'chain_available': bool,
            'reason': human-readable explanation,
            'warnings': [str, ...],
        }
    """
    warnings: list[str] = []

    # 1) Try to fetch live options chain
    chains = get_options_chain(symbol, n_expirations=6)
    chain_available = bool(chains)

    if chain_available:
        # Pick expiration closest to horizon_days
        exp_choices = [(exp, days_to_expiry(exp)) for exp in chains.keys()]
        # Filter to expirations >= horizon
        valid_exps = [(e, d) for e, d in exp_choices if d >= horizon_days]
        if not valid_exps:
            valid_exps = exp_choices  # fallback to whatever we have
        chosen_exp, days = min(valid_exps, key=lambda x: abs(x[1] - horizon_days))
        chain = chains[chosen_exp]
        iv_atm = find_atm_iv(chain, spot)
        iv_used = iv_atm if iv_atm and iv_atm > 0 else historical_vol
        iv_source = "market" if iv_atm and iv_atm > 0 else "historical"
    else:
        chain = None
        chosen_exp = None
        days = horizon_days
        iv_used = historical_vol
        iv_source = "historical"
        warnings.append(f"No options chain available for {symbol} — pricing is theoretical (Black-Scholes on historical vol).")

    # 2) Expected move
    em_dollars = expected_move(spot, iv_used, days)
    em_pct = (em_dollars / spot) * 100

    # 3) Strategy selection based on signal + confidence
    is_high_confidence = confidence >= 0.30  # P(up) deviates ≥15% from 50/50

    def get_strikes_and_prices(target_strike: float, opt_type: str) -> tuple[float, float]:
        """Pick the listed strike closest to target, fetch its quote.
        Falls back to BS theoretical price."""
        if chain is not None:
            chain_df = chain["calls"] if opt_type == "call" else chain["puts"]
            available = chain_df["strike"].tolist()
            K = closest_strike(available, target_strike)
            premium = find_quote(chain_df, K)
            if premium is None:
                premium = bs_price(spot, K, days / 365, risk_free_rate, iv_used, opt_type)
                warnings.append(f"No live quote for ${K:.2f} {opt_type} — using Black-Scholes estimate.")
        else:
            K = round(target_strike, 0)  # round to whole dollar
            premium = bs_price(spot, K, days / 365, risk_free_rate, iv_used, opt_type)
        return K, premium

    if signal == "long":
        if is_high_confidence:
            # High conviction long: long ATM call (uncapped reward)
            K, premium = get_strikes_and_prices(spot, "call")
            strat = long_call(spot, K, premium, days)
            reason = "Model has strong bullish conviction. Long call gives uncapped upside if right; max loss = premium."
        else:
            # Lower conviction long: bull call spread (cheaper, defined risk-reward)
            K_long, p_long = get_strikes_and_prices(spot, "call")
            K_short, p_short = get_strikes_and_prices(spot + em_dollars, "call")
            if K_short <= K_long:  # safety: ensure short strike is above long
                K_short = K_long + max(1.0, round(spot * 0.02, 0))
                p_short = bs_price(spot, K_short, days / 365, risk_free_rate, iv_used, "call")
            strat = bull_call_spread(spot, K_long, K_short, p_long, p_short, days)
            reason = "Model is bullish but not strongly. Bull call spread reduces cost vs naked call by capping reward at the +1σ expected-move strike."

    elif signal == "short":
        if is_high_confidence:
            K, premium = get_strikes_and_prices(spot, "put")
            strat = long_put(spot, K, premium, days)
            reason = "Model has strong bearish conviction. Long put profits as price falls; max loss = premium."
        else:
            K_long, p_long = get_strikes_and_prices(spot, "put")
            K_short, p_short = get_strikes_and_prices(spot - em_dollars, "put")
            if K_short >= K_long:
                K_short = K_long - max(1.0, round(spot * 0.02, 0))
                p_short = bs_price(spot, K_short, days / 365, risk_free_rate, iv_used, "put")
            strat = bear_put_spread(spot, K_long, K_short, p_long, p_short, days)
            reason = "Model is bearish but not strongly. Bear put spread reduces cost vs naked put by capping reward at the -1σ expected-move strike."

    else:  # flat
        # Iron condor: sell premium on both sides, profit if range-bound
        K_call_short, p_call_short = get_strikes_and_prices(spot + em_dollars, "call")
        K_call_long, p_call_long = get_strikes_and_prices(spot + 2 * em_dollars, "call")
        K_put_short, p_put_short = get_strikes_and_prices(spot - em_dollars, "put")
        K_put_long, p_put_long = get_strikes_and_prices(spot - 2 * em_dollars, "put")
        # Sanity: make sure strikes are distinct
        if K_call_long <= K_call_short or K_put_long >= K_put_short:
            warnings.append("Iron condor strikes collapsed — consider a different expiration.")
        strat = iron_condor(
            spot, K_put_long, K_put_short, K_call_short, K_call_long,
            p_put_long, p_put_short, p_call_short, p_call_long, days,
        )
        reason = "Model has no directional edge. Iron condor sells premium on both wings — profitable if price stays within ±1σ of current."

    strat.iv_used = iv_used

    # Sanity check: flag suspiciously low premiums (likely stale quotes)
    if signal in ("long", "short"):
        if strat.name in ("Long call", "Long put"):
            if strat.net_debit < spot * 0.001:
                warnings.append(
                    f"Premium ${strat.net_debit:.2f} is suspiciously low "
                    f"(<0.1% of spot). Check live quotes — may be stale data."
                )
        elif "spread" in strat.name.lower():
            spread_width = abs(strat.legs[0].strike - strat.legs[1].strike)
            if 0 < strat.net_debit < spread_width * 0.05:
                warnings.append(
                    f"Net debit ${strat.net_debit:.2f} is <5% of ${spread_width:.2f} spread width. "
                    f"Either an arbitrage (verify), or stale quotes — confirm before trading."
                )

    return {
        "strategy": strat,
        "expiration": chosen_exp,
        "spot": spot,
        "expected_move_dollars": em_dollars,
        "expected_move_pct": em_pct,
        "iv_used": iv_used,
        "iv_source": iv_source,
        "chain_available": chain_available,
        "reason": reason,
        "warnings": warnings,
        "days_to_expiry": days,
    }


def payoff_diagram(strat: Strategy, spot: float, n_points: int = 100, range_pct: float = 0.30) -> pd.DataFrame:
    """DataFrame for plotting the strategy's P&L at expiration. Two columns: price, pnl."""
    s_min = spot * (1 - range_pct)
    s_max = spot * (1 + range_pct)
    prices = np.linspace(s_min, s_max, n_points)
    pnls = np.array([strat.payoff(s) for s in prices])
    return pd.DataFrame({"price": prices, "pnl": pnls}).set_index("price")
