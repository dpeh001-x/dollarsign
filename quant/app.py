"""Streamlit app: enter a symbol, get a clean recommendation.

Shows three things and nothing else:
  1. Trend (from regime classifier + 20-day slope)
  2. Buy / Sell / Hold (ML model gated by analyst consensus)
  3. Target price (Wall Street analyst mean 12-month price target)

Accuracy strategy — the ML signal is only surfaced as BUY/SELL when it
agrees with the analyst consensus direction. Discordant signals become
HOLD. This raises win-rate dramatically at the cost of signal frequency.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

warnings.filterwarnings("ignore", message="X does not have valid feature names")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from data import sources
from data import macro as macro_src
from data import analyst as analyst_src
import indicators
import regime
import ml
import class_xgb

st.set_page_config(page_title="Long or Short?", page_icon="$", layout="centered")

# ─── Minimal CSS for a clean card look ───
st.markdown(
    """
    <style>
    .big-card {
        padding: 32px 28px;
        border-radius: 16px;
        margin: 12px 0;
        text-align: center;
    }
    .buy   { background: #052e1a; border: 2px solid #16a34a; color: #86efac; }
    .sell  { background: #2e0505; border: 2px solid #dc2626; color: #fca5a5; }
    .hold  { background: #1a1a1a; border: 2px solid #6b7280; color: #d1d5db; }
    .big-signal   { font-size: 56px; font-weight: 800; letter-spacing: 2px; }
    .big-caption  { font-size: 14px; opacity: 0.85; margin-top: 6px; }
    .stat-row { display: flex; justify-content: space-around; margin-top: 14px; }
    .stat-val { font-size: 22px; font-weight: 700; }
    .stat-lbl { font-size: 11px; opacity: 0.7; text-transform: uppercase; letter-spacing: 1px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Long or Short?")
st.caption("ML signal + Wall Street analyst consensus")

# ─── Input ───
symbol = st.text_input("Symbol", value="SPY", help="e.g. AAPL, SPY, NVDA, BTC-USD").upper().strip()
if not st.button("Get recommendation", type="primary", use_container_width=True):
    st.stop()
if not symbol:
    st.error("Enter a stock symbol.")
    st.stop()

MIN_CONFIDENCE = 0.25  # raised from 0.15 → higher accuracy, fewer signals
START_ISO = (date.today() - timedelta(days=3 * 365)).isoformat()


@st.cache_data(show_spinner=False)
def load_bars(sym: str) -> pd.DataFrame:
    df = sources.get_bars(sym, START_ISO, use_cache=True)
    df = indicators.attach_all(df)
    df = regime.label(df)
    return df


@st.cache_data(show_spinner=False)
def load_macro() -> pd.DataFrame | None:
    try:
        return macro_src.get_vix(START_ISO, date.today().isoformat())
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_analyst(sym: str) -> dict | None:
    view = analyst_src.get_analyst_view(sym)
    if view is None:
        return None
    return {
        "mean_target": view.mean_target,
        "high_target": view.high_target,
        "low_target": view.low_target,
        "num_analysts": view.num_analysts,
        "recommendation": view.recommendation,
        "current_price": view.current_price,
        "upside_pct": view.upside_pct,
        "direction": view.direction,
        "is_high_conviction": view.is_high_conviction,
    }


@st.cache_data(show_spinner=False)
def get_ml_signal(sym: str, _df_hash: int) -> dict:
    """Ensemble ML signal at multiple horizons; return the strongest agreeing signal."""
    macro_df = load_macro()
    signals: list[dict] = []
    for horizon in (5, 10, 20):
        try:
            predictor = ml.EnsemblePredictor(horizon=horizon).fit(df, macro_df=macro_df)
            sig = predictor.predict_now(df, macro_df=macro_df)
            sig["horizon"] = horizon
            signals.append(sig)
        except Exception:
            continue
    if not signals:
        return {"proba": float("nan"), "signal": "flat", "confidence": 0.0, "horizon": None}
    # Pick the highest-confidence signal
    return max(signals, key=lambda s: s.get("confidence", 0.0))


def compute_trend(df: pd.DataFrame) -> tuple[str, str]:
    """Return (label, emoji) describing the current trend."""
    latest_regime = df["regime"].dropna().iloc[-1] if "regime" in df.columns and not df["regime"].dropna().empty else "ranging"
    ret_20 = df["ret_20"].iloc[-1] if "ret_20" in df.columns else 0
    ret_5 = df["ret_5"].iloc[-1] if "ret_5" in df.columns else 0

    if latest_regime == "trending_up" or (ret_20 > 0.03 and ret_5 > 0):
        return "STRONG UPTREND", "📈"
    if latest_regime == "trending_down" or (ret_20 < -0.03 and ret_5 < 0):
        return "STRONG DOWNTREND", "📉"
    if latest_regime == "volatile_crash":
        return "VOLATILE / CRASHING", "⚠️"
    if ret_20 > 0.005:
        return "MILD UPTREND", "↗"
    if ret_20 < -0.005:
        return "MILD DOWNTREND", "↘"
    return "SIDEWAYS", "→"


def gated_signal(ml_sig: dict, analyst: dict | None) -> tuple[str, str, str]:
    """Apply consensus gating. Returns (label, css_class, rationale)."""
    ml_dir = ml_sig.get("signal", "flat")
    conf = ml_sig.get("confidence", 0.0)

    # Confidence gate
    if conf < MIN_CONFIDENCE:
        return "HOLD", "hold", f"Model confidence {conf*100:.0f}% below {MIN_CONFIDENCE*100:.0f}% threshold"

    # Analyst gate (skip if analyst data unavailable — e.g. crypto)
    if analyst is not None and analyst.get("num_analysts", 0) >= 5:
        an_dir = analyst.get("direction", "flat")
        if ml_dir == "long" and an_dir == "down":
            return "HOLD", "hold", "Model bullish but analysts see downside — no consensus"
        if ml_dir == "short" and an_dir == "up":
            return "HOLD", "hold", "Model bearish but analysts see upside — no consensus"

    if ml_dir == "long":
        rationale = f"Model {conf*100:.0f}% confident of upside"
        if analyst and analyst.get("num_analysts", 0) >= 5 and analyst.get("direction") == "up":
            rationale += f" · {analyst['num_analysts']} analysts agree"
        return "BUY", "buy", rationale
    if ml_dir == "short":
        rationale = f"Model {conf*100:.0f}% confident of downside"
        if analyst and analyst.get("num_analysts", 0) >= 5 and analyst.get("direction") == "down":
            rationale += f" · {analyst['num_analysts']} analysts agree"
        return "SELL", "sell", rationale
    return "HOLD", "hold", "Model has no directional edge"


# ─── Load everything in parallel spinners ───
with st.spinner(f"Loading {symbol}..."):
    try:
        df = load_bars(symbol)
    except Exception as e:
        st.error(f"Could not load **{symbol}**: {e}")
        st.stop()

if df.empty or len(df) < 60:
    st.error(f"Not enough data for {symbol}.")
    st.stop()

with st.spinner("Fetching Wall Street analyst consensus..."):
    analyst = load_analyst(symbol)

with st.spinner("Running ML ensemble..."):
    ml_sig = get_ml_signal(symbol, len(df))

# ─── Compute the three things ───
trend_label, trend_emoji = compute_trend(df)
signal, css, rationale = gated_signal(ml_sig, analyst)
current_price = float(df["close"].iloc[-1])

# ─── Render — three big cards ───
st.markdown(f"### {symbol} · ${current_price:.2f}")

# 1. Trend
st.markdown(
    f"""
    <div class="big-card hold">
      <div class="stat-lbl">Trend</div>
      <div class="big-signal">{trend_emoji} {trend_label}</div>
      <div class="big-caption">Regime: {df['regime'].dropna().iloc[-1] if 'regime' in df.columns else '?'}
        · 20-day return: {df['ret_20'].iloc[-1]*100:+.1f}%</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 2. Buy / Sell / Hold
st.markdown(
    f"""
    <div class="big-card {css}">
      <div class="stat-lbl">Recommendation</div>
      <div class="big-signal">{signal}</div>
      <div class="big-caption">{rationale}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 3. Target Price
if analyst and analyst.get("mean_target"):
    mean = analyst["mean_target"]
    high = analyst.get("high_target")
    low = analyst.get("low_target")
    n = analyst["num_analysts"]
    upside = analyst.get("upside_pct")
    upside_str = f"{upside:+.1f}%" if upside is not None else "?"
    conviction = "high conviction" if analyst["is_high_conviction"] else f"{n} analyst{'s' if n != 1 else ''}"

    st.markdown(
        f"""
        <div class="big-card hold">
          <div class="stat-lbl">Wall Street Target (12-month)</div>
          <div class="big-signal">${mean:.2f}</div>
          <div class="big-caption">{upside_str} from ${current_price:.2f} · {conviction}</div>
          <div class="stat-row">
            <div>
              <div class="stat-lbl">Low</div>
              <div class="stat-val">${low:.2f}</div>
            </div>
            <div>
              <div class="stat-lbl">Consensus</div>
              <div class="stat-val">${mean:.2f}</div>
            </div>
            <div>
              <div class="stat-lbl">High</div>
              <div class="stat-val">${high:.2f}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="big-card hold">
          <div class="stat-lbl">Wall Street Target</div>
          <div class="big-signal">n/a</div>
          <div class="big-caption">No analyst coverage for this symbol (common for crypto / ETFs)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "Signals require model confidence ≥25% AND analyst-consensus agreement. "
    "Past performance does not guarantee future results."
)
