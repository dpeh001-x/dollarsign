"""Monte Carlo probability-cone forecasting — the honest crystal ball.

No one can read the future. What CAN be done: simulate thousands of
plausible price paths whose drift is tilted by the model's *measured*
edge (live-calibrated win rate, see signal_log.py) and whose dispersion
matches realized volatility. The output is a probability cone: "90% of
simulated futures land inside this fan."

Drift construction (median-calibrated — exact coherence with the win rate):
  Paths are simulated on log-returns WITHOUT the GBM Ito correction, i.e.
  log increments ~ N(drift_d, sigma_d^2). Over the model's 10-trading-day
  horizon the total drift mu_h = sigma_h * Phi^-1(p) (sign flipped for
  shorts), which makes P(10-day directional move > 0) EXACTLY p and keeps
  the median path flat after day 10 (the model predicts nothing beyond
  its horizon). The cone is therefore median-calibrated: the p50 line is
  the literal "half of futures above, half below" path.
  win_prob is clamped to [0.02, 0.98] so extreme calibrations tilt hard
  instead of silently producing infinite/zero drift.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

EDGE_HORIZON_DAYS = 10
PERCENTILES = [5, 25, 50, 75, 95]


def simulate_cone(
    spot: float,
    daily_vol: float,
    win_prob: float,
    direction: str,           # 'long' | 'short' | 'flat'
    days: int = 30,
    n_paths: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate GBM paths with edge-tilted drift. Returns DataFrame indexed
    by day (0..days) with columns p5, p25, p50, p75, p95 (price levels)."""
    rng = np.random.default_rng(seed)
    sigma_h = daily_vol * np.sqrt(EDGE_HORIZON_DAYS)

    if direction in ("long", "short"):
        p = float(np.clip(win_prob, 0.02, 0.98))   # never silently zero the edge
        mu_h = sigma_h * norm.ppf(p)               # median-calibrated horizon drift
        if direction == "short":
            mu_h = -mu_h
    else:
        mu_h = 0.0
    mu_daily_edge = mu_h / EDGE_HORIZON_DAYS

    # Median-calibrated log-return draws (no Ito term): drift only during the
    # edge window, exactly-flat median afterwards, P(10d move) == win_prob.
    drifts = np.array([mu_daily_edge if d < EDGE_HORIZON_DAYS else 0.0 for d in range(days)])
    shocks = rng.normal(0.0, daily_vol, size=(n_paths, days))
    log_paths = np.cumsum(drifts[None, :] + shocks, axis=1)
    prices = spot * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))

    rows = {f"p{p}": np.percentile(prices, p, axis=0) for p in PERCENTILES}
    out = pd.DataFrame(rows)
    out.index.name = "day"
    return out


def cone_summary(cone: pd.DataFrame, spot: float, at_day: int = 10) -> dict:
    """Headline numbers at the model horizon."""
    at_day = min(at_day, len(cone) - 1)
    row = cone.iloc[at_day]
    return {
        "day": at_day,
        "median": float(row["p50"]),
        "median_pct": float(row["p50"] / spot - 1) * 100,
        "p5_pct": float(row["p5"] / spot - 1) * 100,
        "p95_pct": float(row["p95"] / spot - 1) * 100,
    }


def plot_cone(
    ticker: str,
    history: pd.Series,        # recent closes (tail ~60 bars)
    cone: pd.DataFrame,
    direction: str,
    win_prob: float,
    out_path: Path | str | None = None,
) -> Path:
    """Render history + probability cone to PNG. Returns the path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    hist = history.tail(60)
    last_date = hist.index[-1]
    fdates = pd.bdate_range(last_date, periods=len(cone), freq="B")

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")

    ax.plot(hist.index, hist.values, color="#E8E6E1", lw=1.6, label="history")

    dir_color = {"long": "#00E68A", "short": "#FF4757", "flat": "#FF9F43"}.get(direction, "#FF9F43")
    ax.fill_between(fdates, cone["p5"], cone["p95"], color=dir_color, alpha=0.13, label="90% of simulated futures")
    ax.fill_between(fdates, cone["p25"], cone["p75"], color=dir_color, alpha=0.25, label="50% of simulated futures")
    ax.plot(fdates, cone["p50"], color=dir_color, lw=2.0, ls="--", label="median path")
    ax.axvline(fdates[min(EDGE_HORIZON_DAYS, len(fdates) - 1)], color="#FFD93D", lw=0.8, ls=":", alpha=0.7)
    ax.text(fdates[min(EDGE_HORIZON_DAYS, len(fdates) - 1)], ax.get_ylim()[0], " edge horizon",
            color="#FFD93D", fontsize=8, va="bottom")

    ax.set_title(
        f"{ticker} — Monte Carlo probability cone ({direction.upper()}, calibrated win {win_prob:.0%})",
        color="#E8E6E1", fontsize=12,
    )
    ax.tick_params(colors="#9B99A1")
    for spine in ax.spines.values():
        spine.set_color("#2A2B38")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(color="#2A2B38", lw=0.4, alpha=0.6)
    leg = ax.legend(facecolor="#1A1B24", edgecolor="#2A2B38", labelcolor="#E8E6E1", fontsize=8)
    fig.tight_layout()

    if out_path is None:
        out_path = REPORTS_DIR / f"cone_{ticker.replace('-','_')}.png"
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def forecast_ticker(ticker: str, direction: str, conf: float, days: int = 30) -> tuple[pd.DataFrame, dict, Path]:
    """End-to-end: load data, calibrated win rate, simulate, plot.
    Returns (cone, summary, png_path)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from datetime import date, timedelta
    import indicators
    import signal_log
    from data import sources

    df = sources.get_bars(ticker, (date.today() - timedelta(days=2 * 365)).isoformat(), use_cache=True)
    df = indicators.attach_all(df)
    spot = float(df["close"].iloc[-1])
    daily_vol = float(df["vol_20"].iloc[-1]) / np.sqrt(252)
    if not np.isfinite(daily_vol) or daily_vol <= 0:
        daily_vol = 0.012

    win = signal_log.get_calibrated_win(conf) if direction in ("long", "short") else 0.5
    cone = simulate_cone(spot, daily_vol, win, direction, days=days)
    summary = cone_summary(cone, spot)
    png = plot_cone(ticker, df["close"], cone, direction, win)
    return cone, summary, png


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--direction", default="flat", choices=["long", "short", "flat"])
    parser.add_argument("--conf", type=float, default=0.0)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    cone, s, png = forecast_ticker(args.ticker, args.direction, args.conf, args.days)
    print(f"{args.ticker} {args.direction} — median day-{s['day']}: {s['median_pct']:+.1f}%  "
          f"(90% cone: {s['p5_pct']:+.1f}% to {s['p95_pct']:+.1f}%)")
    print(f"chart: {png}")
