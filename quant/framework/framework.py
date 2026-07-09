"""Multi-factor framework orchestrator.

Wires the four models together into one daily trade plan:

  1. Regime filter decides WHICH sleeves may trade and at what size.
  2. Cross-sectional momentum ranks the basket → longs / shorts / dip-buys.
  3. Pairs scanner emits market-neutral spread trades (regime-independent).
  4. Dynamic risk sizes every candidate and enforces the portfolio heat cap.

The output is a plain dict — printable by the CLI, serializable to JSON,
consumable by anything else.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import sources  # noqa: E402
import indicators  # noqa: E402

from framework.regime_filter import market_regime, RegimeState  # noqa: E402
from framework import xsect_momentum as xs  # noqa: E402
from framework.pairs import scan_pairs, DEFAULT_PAIRS  # noqa: E402
from framework import risk as risk_mod  # noqa: E402

logger = logging.getLogger(__name__)

# Default basket: liquid US large/mid caps across sectors + HK mega-caps.
# The individual trader's edge is agility — swap in the smaller names you
# actually follow; the ranking math doesn't care.
DEFAULT_UNIVERSE: list[str] = [
    # US mega/large
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "AVGO",
    "NFLX", "CRM", "JPM", "V", "MA", "HD", "LOW", "KO", "PEP", "XOM", "CVX",
    "UNH", "LLY", "COST", "ORCL", "QCOM",
    # Liquid mid-caps / higher-beta names (the agility zone)
    "PLTR", "DKNG", "ROKU", "ETSY", "SOFI",
    # HK
    "0700.HK", "9988.HK", "3690.HK", "1299.HK", "0005.HK",
]


@dataclass
class FrameworkConfig:
    universe: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    pairs: list[tuple[str, str]] = field(default_factory=lambda: list(DEFAULT_PAIRS))
    benchmark: str = "SPY"
    account_equity: float = 10_000.0
    risk_pct: float = 1.0          # % of equity risked per trade
    max_heat_pct: float = 5.0      # total open risk cap across the book
    vix_target: float = 20.0
    top_n: int = 5                 # momentum longs
    bottom_n: int = 3              # momentum shorts (needs allow_shorts)
    dip_n: int = 3                 # mean-reversion buys
    allow_shorts: bool = False     # borrow costs/availability are real; opt-in
    lookback_days: int = 460       # calendar days of history to fetch


def _load_closes(symbols: list[str], start: str) -> tuple[dict[str, pd.Series], dict[str, float]]:
    """Fetch bars for every symbol; return ({sym: close series}, {sym: latest ATR})."""
    closes: dict[str, pd.Series] = {}
    atrs: dict[str, float] = {}
    for sym in symbols:
        try:
            bars = sources.get_bars(sym, start, use_cache=True)
            closes[sym] = bars["close"]
            atrs[sym] = float(indicators.atr(bars, 14).iloc[-1])
        except Exception as e:
            logger.warning("skip %s: %s", sym, e)
    return closes, atrs


def _latest_vix(start: str) -> float | None:
    try:
        from data import macro as macro_src
        vix = macro_src.get_vix(start, date.today().isoformat())
        return float(vix["vix_level"].iloc[-1]) if not vix.empty else None
    except Exception as e:
        logger.warning("VIX unavailable: %s", e)
        return None


def daily_plan(cfg: FrameworkConfig | None = None) -> dict:
    """Run all four models and produce today's sized trade plan."""
    cfg = cfg or FrameworkConfig()
    start = (date.today() - timedelta(days=cfg.lookback_days)).isoformat()

    # ── Model 1: regime ──
    regime = market_regime(cfg.benchmark, start=start)

    # ── Data for models 2-4 ──
    pair_syms = sorted({s for p in cfg.pairs for s in p})
    all_syms = sorted(set(cfg.universe) | set(pair_syms))
    closes, atrs = _load_closes(all_syms, start)
    vix = _latest_vix(start)

    # ── Model 2: cross-sectional table + sleeves, gated by regime ──
    table = xs.rank_universe({s: c for s, c in closes.items() if s in cfg.universe})

    momentum_buys: list[str] = []
    dip_buys: list[str] = []
    shorts: list[str] = []
    if regime.state == "risk_on":
        momentum_buys = xs.momentum_longs(table, cfg.top_n)
        dip_buys = xs.dip_buys(table, cfg.dip_n)
    elif regime.state == "neutral":
        dip_buys = xs.dip_buys(table, cfg.dip_n)          # mean-reversion only, half size
    else:  # risk_off — the filter that stops dip-buying a bear market
        if cfg.allow_shorts:
            shorts = xs.momentum_shorts(table, cfg.bottom_n)

    # ── Model 3: pairs (market-neutral — runs in every regime) ──
    pair_signals = scan_pairs(closes, cfg.pairs)
    active_pairs = [p for p in pair_signals if p.signal in ("long_a_short_b", "short_a_long_b")]

    # ── Model 4: size everything, then heat-cap the book ──
    size_scale = 1.0 if regime.full_size else 0.5
    candidates: list[risk_mod.SizedPosition] = []

    def _size(sym: str, side: str, scale: float, tag: str):
        if sym not in closes or sym not in atrs:
            return
        pos = risk_mod.size_position(
            sym, side, float(closes[sym].iloc[-1]), atrs[sym],
            cfg.account_equity, cfg.risk_pct, vix, cfg.vix_target,
            size_scale=scale,
        )
        if pos:
            pos.note = (tag + ("; " + pos.note if pos.note else ""))
            candidates.append(pos)

    for sym in momentum_buys:
        _size(sym, "long", size_scale, "momentum")
    for sym in dip_buys:
        _size(sym, "long", size_scale * 0.5 if regime.state == "neutral" else size_scale, "dip-buy (RSI<30, above 200d)")
    for sym in shorts:
        _size(sym, "short", size_scale, "momentum short (below 200d)")
    for p in active_pairs:
        long_leg, short_leg = (p.a, p.b) if p.signal == "long_a_short_b" else (p.b, p.a)
        _size(long_leg, "long", 0.5, f"pair {p.a}/{p.b} z={p.zscore:+.1f}")
        _size(short_leg, "short", 0.5, f"pair {p.a}/{p.b} z={p.zscore:+.1f}")

    accepted, deferred = risk_mod.apply_heat_cap(candidates, cfg.account_equity, cfg.max_heat_pct)

    return {
        "as_of": date.today().isoformat(),
        "regime": {
            "state": regime.state, "detail": regime.detail,
            "benchmark": regime.benchmark, "allow_new_longs": regime.allow_new_longs,
        },
        "vix": vix,
        "ranks": table.reset_index().to_dict(orient="records") if not table.empty else [],
        "momentum_buys": momentum_buys,
        "dip_buys": dip_buys,
        "shorts": shorts,
        "pairs": [vars(p) for p in pair_signals],
        "trades": [vars(p) for p in accepted],
        "deferred": [vars(p) for p in deferred],
        "open_risk_dollars": round(sum(p.risk_dollars for p in accepted), 2),
        "config": {
            "equity": cfg.account_equity, "risk_pct": cfg.risk_pct,
            "max_heat_pct": cfg.max_heat_pct, "allow_shorts": cfg.allow_shorts,
        },
    }
