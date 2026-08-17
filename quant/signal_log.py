"""Signal journal + live calibration — the self-improvement loop.

Every screen run appends its signals here. Once a signal's horizon has
fully elapsed, `score_matured()` fetches what actually happened and marks
it hit/miss. `calibrate()` then blends the LIVE track record with the
backtest priors to produce an updated confidence→win-rate map, which the
forecast engine and EV math consume.

This is the honest version of "an algorithm that improves": the system's
probability estimates converge toward its measured real-world accuracy
instead of staying frozen at backtest values.

Files (in quant/journal/):
    signals.csv        one row per (run_date, ticker) signal
    calibration.json   current conf-band → win-rate map (live-blended)
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

JOURNAL_DIR = Path(__file__).resolve().parent / "journal"
JOURNAL_DIR.mkdir(exist_ok=True)
SIGNALS_CSV = JOURNAL_DIR / "signals.csv"
CALIBRATION_JSON = JOURNAL_DIR / "calibration.json"

HORIZON_TRADING_DAYS = 10
MATURITY_CALENDAR_DAYS = 16  # ~10 trading days + weekend buffer

# Confidence bands and their backtest-prior win rates (from reliability.py
# and hypothetical runs). Priors are deliberately conservative and carry a
# pseudo-count so early live data doesn't whipsaw the calibration.
BANDS: list[tuple[float, float, str]] = [
    (0.00, 0.15, "0-15%"),
    (0.15, 0.30, "15-30%"),
    (0.30, 0.45, "30-45%"),
    (0.45, 0.60, "45-60%"),
    (0.60, 1.01, "60%+"),
]
PRIOR_WIN = {"0-15%": 0.52, "15-30%": 0.56, "30-45%": 0.58, "45-60%": 0.60, "60%+": 0.62}
PRIOR_WEIGHT = 40  # pseudo-trades per band backing the prior


def band_of(conf: float) -> str:
    if conf < BANDS[0][0]:          # garbage/negative conf → LOWEST band, not highest
        return BANDS[0][2]
    for lo, hi, name in BANDS:
        if lo <= conf < hi:
            return name
    return BANDS[-1][2]


def _load() -> pd.DataFrame:
    if SIGNALS_CSV.exists():
        # round_trip: keep floats byte-stable across load/save cycles
        df = pd.read_csv(SIGNALS_CSV, parse_dates=["run_date", "scored_date"],
                         float_precision="round_trip")
        # pandas 3 forbids assigning bool into a float64 (all-NaN) column —
        # store hit as float 1.0/0.0 and force the dtype up front.
        if "hit" in df.columns:
            df["hit"] = df["hit"].astype("float64")
        return df
    return pd.DataFrame(columns=[
        "run_date", "ticker", "direction", "proba", "conf", "spot",
        "horizon_days", "status", "outcome_return", "hit", "scored_date",
    ])


def _save(df: pd.DataFrame) -> None:
    df.to_csv(SIGNALS_CSV, index=False)


def log_signals(results: list) -> int:
    """Append today's screen results (list of objects with .ticker/.signal/
    .proba/.confidence/.last_close, or equivalent dicts). Skips flats and
    duplicates for the same (run_date, ticker). Returns rows added."""
    df = _load()
    today = pd.Timestamp(date.today())
    added = 0
    for r in results:
        # Per-row isolation: one malformed result must not lose the batch.
        try:
            get = (lambda k, rr=r: getattr(rr, k, None) if not isinstance(rr, dict) else rr.get(k))
            signal = get("signal")
            if signal not in ("long", "short"):
                continue
            ticker = get("ticker")
            dup = df[(df["ticker"] == ticker) & (df["run_date"] == today)]
            if len(dup) > 0:
                continue
            conf = get("confidence") if get("confidence") is not None else get("conf")
            spot = get("last_close") if get("last_close") is not None else get("spot")
            df.loc[len(df)] = {
                "run_date": today, "ticker": ticker, "direction": signal,
                "proba": float(get("proba")), "conf": float(conf),
                "spot": float(spot) if spot else np.nan,
                "horizon_days": HORIZON_TRADING_DAYS, "status": "open",
                "outcome_return": np.nan, "hit": np.nan, "scored_date": pd.NaT,
            }
            added += 1
        except (TypeError, ValueError) as e:
            logger.error("log_signals: skipping malformed result %r: %s", r, e)
    if added:
        _save(df)
    return added


def score_matured() -> int:
    """Score open signals whose maturity window has passed. Uses the close
    of the first bar ON/AFTER run_date as entry and the close HORIZON
    trading bars later as exit. No look-ahead: only fully-elapsed signals
    are scored. Returns number scored."""
    from data import sources  # local import to avoid cycles

    df = _load()
    if df.empty:
        return 0
    now = pd.Timestamp(date.today())
    open_mask = (df["status"] == "open") & (df["run_date"] <= now - pd.Timedelta(days=MATURITY_CALENDAR_DAYS))
    scored = 0
    failures = 0
    for idx in df[open_mask].index:
        row = df.loc[idx]
        try:
            start = (row["run_date"] - pd.Timedelta(days=10)).date().isoformat()
            bars = sources.get_bars(row["ticker"], start, use_cache=True)
            bars = bars[~bars.index.duplicated(keep="first")]
            # tz guard: normalize both sides to UTC regardless of source
            if bars.index.tz is None:
                bars.index = bars.index.tz_localize("UTC")
            run_ts = row["run_date"]
            run_ts = run_ts.tz_localize("UTC") if run_ts.tzinfo is None else run_ts.tz_convert("UTC")
            entry_pos_candidates = bars.index[bars.index >= run_ts]
            if len(entry_pos_candidates) == 0:
                continue
            entry_i = bars.index.get_loc(entry_pos_candidates[0])
            exit_i = entry_i + int(row["horizon_days"])
            if exit_i >= len(bars):
                continue  # horizon not fully elapsed in trading days yet
            entry_px = float(bars["close"].iloc[entry_i])
            exit_px = float(bars["close"].iloc[exit_i])
            ret = exit_px / entry_px - 1.0
            directional = ret if row["direction"] == "long" else -ret
            df.loc[idx, "outcome_return"] = round(directional, 5)
            df.loc[idx, "hit"] = 1.0 if directional > 0 else 0.0  # float, pandas-3 safe
            df.loc[idx, "status"] = "scored"
            df.loc[idx, "scored_date"] = now
            scored += 1
        except Exception as e:
            failures += 1
            logger.error("scoring %s/%s FAILED: %s: %s",
                         row["ticker"], row["run_date"], type(e).__name__, e)
    if failures:
        logger.error("score_matured: %d signal(s) failed to score — investigate, "
                     "they will be retried next run", failures)
    if scored:
        _save(df)
    return scored


def calibrate() -> dict:
    """Blend live scored results with backtest priors per confidence band.
    posterior_win = (prior_win*PRIOR_WEIGHT + live_wins) / (PRIOR_WEIGHT + live_n)
    Writes calibration.json and returns it."""
    df = _load()
    scored = df[df["status"] == "scored"]
    out = {"updated": datetime.now(timezone.utc).isoformat(), "bands": {}}
    for lo, hi, name in BANDS:
        live = scored[(scored["conf"] >= lo) & (scored["conf"] < hi)]
        live_n = int(len(live))
        live_wins = int(live["hit"].sum()) if live_n else 0
        post = (PRIOR_WIN[name] * PRIOR_WEIGHT + live_wins) / (PRIOR_WEIGHT + live_n)
        out["bands"][name] = {
            "prior_win": PRIOR_WIN[name],
            "live_n": live_n,
            "live_win": round(live_wins / live_n, 4) if live_n else None,
            "posterior_win": round(post, 4),
            "avg_move": round(float(live["outcome_return"].abs().mean()), 5) if live_n else None,
        }
    CALIBRATION_JSON.write_text(json.dumps(out, indent=2))
    return out


def get_calibrated_win(conf: float) -> float:
    """Current best win-rate estimate for a signal at this confidence."""
    if CALIBRATION_JSON.exists():
        cal = json.loads(CALIBRATION_JSON.read_text())
        return float(cal["bands"][band_of(conf)]["posterior_win"])
    return PRIOR_WIN[band_of(conf)]


def journal_stats() -> dict:
    df = _load()
    scored = df[df["status"] == "scored"]
    return {
        "total_signals": int(len(df)),
        "open": int((df["status"] == "open").sum()),
        "scored": int(len(scored)),
        "live_win_rate": round(float(scored["hit"].mean()), 4) if len(scored) else None,
        "avg_outcome": round(float(scored["outcome_return"].mean()), 5) if len(scored) else None,
    }


if __name__ == "__main__":
    print("Journal:", journal_stats())
    n = score_matured()
    print(f"Scored {n} matured signals")
    cal = calibrate()
    for name, b in cal["bands"].items():
        live = f"{b['live_win']:.0%} on {b['live_n']}" if b["live_n"] else "no live data"
        print(f"  {name:>7s}: posterior win {b['posterior_win']:.1%}  (prior {b['prior_win']:.0%}, live: {live})")
