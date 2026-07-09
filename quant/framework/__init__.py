"""Multi-factor momentum + mean-reversion framework for individual accounts.

Four composable models:
  regime_filter   — market state gate (risk_on / neutral / risk_off)
  xsect_momentum  — cross-sectional ranking + RSI mean-reversion overlay
  pairs           — statistical arbitrage on correlated pairs (z-score)
  risk            — VIX-scaled fixed-fractional sizing + portfolio heat cap

Entry point: framework.daily_plan(FrameworkConfig(...)) or the CLI:
    python framework_cli.py --equity 25000 --risk 1.0
"""
from framework.regime_filter import market_regime, classify_regime, RegimeState  # noqa: F401
from framework.xsect_momentum import rank_universe, momentum_longs, momentum_shorts, dip_buys  # noqa: F401
from framework.pairs import pair_signal, scan_pairs, PairSignal, DEFAULT_PAIRS  # noqa: F401
from framework.risk import size_position, apply_heat_cap, vix_scalar, SizedPosition  # noqa: F401
from framework.framework import daily_plan, FrameworkConfig, DEFAULT_UNIVERSE  # noqa: F401
