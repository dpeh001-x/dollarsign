"""Tiered signal classification + structural position sizing.

Replaces the old binary "≥75% confidence or FLAT" mapping with a 7-tier ladder:
  STRONG_SHORT < SHORT < LEAN_SHORT < FLAT < LEAN_LONG < LONG < STRONG_LONG

The base tier comes from the model's P(up). It is then *upgraded* one notch
when the trend + momentum tracker AGREES with the model's direction, or
*downgraded* one notch when they DISAGREE. Confluence cannot flip direction —
it only nudges severity along the ladder, never across FLAT.

Conviction (1–5 ★) is a separate metric combining:
  • model probability distance from 0.5
  • technical confluence
  • ADX (trend strength dampens conviction in ranging markets)
  • RSI extremes (overbought longs / oversold shorts are downgraded)

Sizing is structural and account-aware:
  • Stop = WIDER of (2.5×ATR) and (recent 20-day swing - 0.5×ATR buffer)
  • TP   = 1:2 R:R when ADX>25 (trending), 1:1 otherwise
  • Risk = base_risk_pct × conviction/5, halved if ATR%>4%, capped at 25%
    of account by position-value
  • Time stop = 3× horizon in trading days
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np


# ─── Tier ladder (0 = strongest short, 6 = strongest long, 3 = flat) ───
TIERS: list[tuple[str, str, str, int]] = [
    # (key,          label,         hex_color, direction)
    ("STRONG_SHORT", "STRONG SHORT", "#FF4757", -1),
    ("SHORT",        "SELL SHORT",   "#FF6B7B", -1),
    ("LEAN_SHORT",   "LEAN SHORT",   "#FFB4BA", -1),
    ("FLAT",         "STAY FLAT",    "#FF9F43",  0),
    ("LEAN_LONG",    "LEAN LONG",    "#7FE5C5",  1),
    ("LONG",         "BUY LONG",     "#00E68A",  1),
    ("STRONG_LONG",  "STRONG BUY",   "#00BFA5",  1),
]

IDX_FLAT = 3
IDX_STRONG_SHORT = 0
IDX_STRONG_LONG = 6


@dataclass
class TradeSignal:
    proba: float                   # P(up), 0..1
    direction: int                 # -1, 0, +1
    tier_idx: int                  # 0..6 index into TIERS
    tier: str                      # tier key, e.g. "STRONG_LONG"
    label: str                     # display label, e.g. "STRONG BUY"
    color: str
    conviction: int                # 1..5
    model_confidence: float        # |P-0.5|*2
    trend_score: float
    momentum_score: float
    confluence: str                # "aligned" | "mixed" | "opposed" | "neutral"
    reasoning: list[str] = field(default_factory=list)


def _proba_to_tier_idx(p: float) -> int:
    """Map P(up) to a tier index using the deadband ladder."""
    if p < 0.35:
        return 0  # STRONG_SHORT
    if p < 0.42:
        return 1  # SHORT
    if p < 0.48:
        return 2  # LEAN_SHORT
    if p <= 0.52:
        return 3  # FLAT
    if p < 0.58:
        return 4  # LEAN_LONG
    if p < 0.65:
        return 5  # LONG
    return 6      # STRONG_LONG


def _confluence(direction: int, trend_score: float, momentum_score: float) -> str:
    """Label the technical agreement with the model's direction."""
    if direction == 0:
        return "neutral"
    aligned_up = trend_score > 0.25 and momentum_score > 0.20
    aligned_dn = trend_score < -0.25 and momentum_score < -0.20
    if direction > 0:
        if aligned_up:
            return "aligned"
        if aligned_dn:
            return "opposed"
        return "mixed"
    if aligned_dn:
        return "aligned"
    if aligned_up:
        return "opposed"
    return "mixed"


def _apply_confluence(base_idx: int, direction: int, confluence: str) -> int:
    """Nudge the tier toward stronger (aligned) or weaker (opposed). Never crosses FLAT."""
    if direction == 0 or confluence == "mixed" or confluence == "neutral":
        return base_idx
    # Define safe lanes (must not cross FLAT via confluence alone)
    if direction > 0:
        lo, hi = IDX_FLAT + 1, IDX_STRONG_LONG  # lanes 4..6
        if confluence == "aligned":
            return min(hi, base_idx + 1)
        if confluence == "opposed":
            return max(lo, base_idx - 1)
    else:
        lo, hi = IDX_STRONG_SHORT, IDX_FLAT - 1  # lanes 0..2
        if confluence == "aligned":
            return max(lo, base_idx - 1)  # going further from FLAT means smaller index for shorts
        if confluence == "opposed":
            return min(hi, base_idx + 1)
    return base_idx


def _conviction(
    proba: float, trend_score: float, momentum_score: float,
    adx: float, rsi: float, direction: int, confluence: str,
) -> int:
    if direction == 0:
        # For FLAT, conviction = "how confidently neutral" — peaks near P=0.5.
        return max(1, 5 - int(round(abs(proba - 0.5) * 20)))

    pdiff = abs(proba - 0.5) * 2  # 0..1
    if pdiff < 0.05:
        base = 1
    elif pdiff < 0.10:
        base = 2
    elif pdiff < 0.16:
        base = 3
    elif pdiff < 0.24:
        base = 4
    else:
        base = 5

    if confluence == "aligned":
        base += 1
    elif confluence == "opposed":
        base -= 1

    if not np.isnan(adx) and adx < 18:
        base -= 1  # ranging market — trend signals less reliable

    if direction > 0 and not np.isnan(rsi) and rsi > 80:
        base -= 1
    elif direction < 0 and not np.isnan(rsi) and rsi < 20:
        base -= 1

    return max(1, min(5, base))


def classify_signal(
    *,
    proba: float,
    trend_score: float,
    momentum_score: float,
    adx: float = float("nan"),
    rsi: float = float("nan"),
) -> TradeSignal:
    """Map model probability + technical context to a tiered TradeSignal."""
    base_idx = _proba_to_tier_idx(proba)
    base_tier = TIERS[base_idx]
    base_direction = base_tier[3]

    confluence = _confluence(base_direction, trend_score, momentum_score)
    final_idx = _apply_confluence(base_idx, base_direction, confluence)
    final_tier = TIERS[final_idx]
    direction = final_tier[3]

    conviction = _conviction(proba, trend_score, momentum_score, adx, rsi, direction, confluence)

    reasoning: list[str] = [
        f"Model P(up) = {proba:.1%} → base tier {base_tier[1]}",
        f"Trend score {trend_score:+.2f}, momentum score {momentum_score:+.2f}",
    ]
    if base_idx != final_idx:
        if confluence == "aligned":
            reasoning.append(f"Technicals AGREE with model — tier upgraded to {final_tier[1]}")
        else:
            reasoning.append(f"Technicals DISAGREE with model — tier downgraded to {final_tier[1]}")
    elif confluence == "mixed" and direction != 0:
        reasoning.append("Trend + momentum mixed — no tier adjustment")
    elif direction == 0 and confluence == "neutral":
        reasoning.append("Model is on the fence; technicals do not justify a directional bet")

    if not np.isnan(adx):
        if adx >= 25 and abs(trend_score) > 0.4:
            reasoning.append(f"ADX {adx:.0f} confirms an established trend")
        elif adx < 18:
            reasoning.append(f"ADX {adx:.0f} = ranging market — directional moves less reliable")

    if direction > 0 and not np.isnan(rsi) and rsi > 80:
        reasoning.append(f"⚠ RSI {rsi:.0f} extremely overbought — late-entry risk, conviction reduced")
    elif direction < 0 and not np.isnan(rsi) and rsi < 20:
        reasoning.append(f"⚠ RSI {rsi:.0f} extremely oversold — late-entry risk for shorts, conviction reduced")

    return TradeSignal(
        proba=proba, direction=direction,
        tier_idx=final_idx, tier=final_tier[0], label=final_tier[1], color=final_tier[2],
        conviction=conviction,
        model_confidence=abs(proba - 0.5) * 2,
        trend_score=trend_score, momentum_score=momentum_score,
        confluence=confluence,
        reasoning=reasoning,
    )


# ─── Position sizing ────────────────────────────────────────────────


@dataclass
class PositionSize:
    direction: int
    entry: float
    stop: float
    take_profit: float
    risk_per_share: float
    reward_per_share: float
    rr_ratio: float
    pct_to_stop: float          # +/- percent from entry to stop
    pct_to_tp: float
    n_shares: int
    position_value: float
    risk_dollar: float
    risk_pct_of_account: float
    effective_risk_pct: float   # base × conviction/5 × vol_cap
    stop_basis: str             # human description
    tp_basis: str
    time_stop_date: str
    horizon_days: int
    notes: list[str] = field(default_factory=list)


def size_position(
    *,
    direction: int,
    spot: float,
    atr: float,
    conviction: int,
    account_size: float,
    base_risk_pct: float = 0.01,   # 1% baseline; scales by conviction/5
    adx: float = float("nan"),
    swing_low_20d: float = float("nan"),
    swing_high_20d: float = float("nan"),
    horizon_days: int = 10,
    last_date: date | None = None,
    max_position_pct: float = 0.25,
    high_vol_threshold: float = 0.04,
) -> PositionSize | None:
    """Return sizing for a long/short setup, or None for flat / invalid inputs."""
    if direction == 0 or spot <= 0 or atr <= 0 or account_size <= 0 or base_risk_pct <= 0:
        return None

    notes: list[str] = []
    atr_pct = atr / spot
    is_long = direction > 0

    # ─── Stop placement: WIDER of ATR-based and structural-swing-based ───
    atr_stop_dist = 2.5 * atr
    if is_long:
        atr_stop = spot - atr_stop_dist
        structural = None
        if not np.isnan(swing_low_20d) and swing_low_20d < spot:
            cand = swing_low_20d - 0.5 * atr
            if (spot - cand) <= 5 * atr:
                structural = cand
        if structural is not None and structural < atr_stop:
            stop = structural
            stop_basis = f"Below 20d swing low ${swing_low_20d:.2f} (−0.5×ATR buffer)"
        else:
            stop = atr_stop
            stop_basis = "2.5× ATR (volatility-based)"
    else:
        atr_stop = spot + atr_stop_dist
        structural = None
        if not np.isnan(swing_high_20d) and swing_high_20d > spot:
            cand = swing_high_20d + 0.5 * atr
            if (cand - spot) <= 5 * atr:
                structural = cand
        if structural is not None and structural > atr_stop:
            stop = structural
            stop_basis = f"Above 20d swing high ${swing_high_20d:.2f} (+0.5×ATR buffer)"
        else:
            stop = atr_stop
            stop_basis = "2.5× ATR (volatility-based)"

    risk_per_share = abs(spot - stop)
    if risk_per_share <= 0:
        return None

    # ─── R:R from regime ───
    if not np.isnan(adx) and adx > 25:
        rr_ratio = 2.0
        tp_basis = f"1:2 R:R — ADX {adx:.0f} trending, let winners run"
    else:
        adx_str = f"{adx:.0f}" if not np.isnan(adx) else "n/a"
        rr_ratio = 1.0
        tp_basis = f"1:1 R:R — ADX {adx_str} ranging/weak, book at parity"

    if is_long:
        take_profit = spot + rr_ratio * risk_per_share
    else:
        take_profit = spot - rr_ratio * risk_per_share
    reward_per_share = abs(take_profit - spot)

    # ─── Conviction-scaled risk, vol cap ───
    conviction_factor = conviction / 5.0
    effective_risk_pct = base_risk_pct * conviction_factor

    if atr_pct > high_vol_threshold:
        effective_risk_pct *= 0.5
        notes.append(f"High vol (ATR {atr_pct*100:.1f}% > {high_vol_threshold*100:.0f}%) — risk halved")

    risk_dollar = account_size * effective_risk_pct
    n_shares = int(risk_dollar / risk_per_share)

    position_value = n_shares * spot
    max_position_value = account_size * max_position_pct
    if position_value > max_position_value:
        n_shares = int(max_position_value / spot)
        position_value = n_shares * spot
        risk_dollar = n_shares * risk_per_share
        notes.append(
            f"Position capped at {max_position_pct*100:.0f}% of account "
            f"(${max_position_value:,.0f}) — risk reduced from sizing target"
        )

    actual_risk_pct = (n_shares * risk_per_share / account_size) if account_size > 0 else 0.0
    pct_to_stop = (stop / spot - 1) * 100
    pct_to_tp = (take_profit / spot - 1) * 100

    # ─── Time stop: 3× horizon trading days → calendar days ───
    trading_days_stop = horizon_days * 3
    calendar_days_stop = int(round(trading_days_stop * 7 / 5))
    if last_date is None:
        last_date = date.today()
    time_stop = (last_date + timedelta(days=calendar_days_stop)).strftime("%b %d, %Y")

    return PositionSize(
        direction=direction,
        entry=spot, stop=stop, take_profit=take_profit,
        risk_per_share=risk_per_share, reward_per_share=reward_per_share,
        rr_ratio=rr_ratio,
        pct_to_stop=pct_to_stop, pct_to_tp=pct_to_tp,
        n_shares=n_shares, position_value=position_value,
        risk_dollar=risk_dollar, risk_pct_of_account=actual_risk_pct,
        effective_risk_pct=effective_risk_pct,
        stop_basis=stop_basis, tp_basis=tp_basis,
        time_stop_date=time_stop, horizon_days=horizon_days,
        notes=notes,
    )
