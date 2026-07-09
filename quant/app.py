"""Streamlit app: pick a stock, get a clean recommendation.

Design copies the patterns of the major finance sites:
  - Yahoo Finance: color-coded rating, analyst gauge, low/avg/high target range
  - Robinhood: minimalist cards, green/red price cues, zero clutter
  - Google Finance: one-tap popular tickers, instant results (no submit button)

Covers US stocks/ETFs, Hong Kong stocks (.HK — type "700" or "0700.HK"),
and crypto (BTC-USD).

Shows three things:
  1. Trend (regime classifier + 20-day slope)
  2. Buy / Sell / Hold (ML model, gated by verified edge + trend + analysts)
  3. Target price (Wall Street analyst consensus with gauge)

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

import altair as alt
import pandas as pd
import streamlit as st

from data import sources
from data import macro as macro_src
from data import analyst as analyst_src
import indicators
import regime
import ml
import backtest

st.set_page_config(page_title="Long or Short?", page_icon="$", layout="centered")

# ─── CSS: Robinhood-style minimal cards ───
st.markdown(
    """
    <style>
    .big-card {
        padding: 28px 24px;
        border-radius: 16px;
        margin: 10px 0;
        text-align: center;
    }
    .buy   { background: #052e1a; border: 2px solid #16a34a; color: #86efac; }
    .sell  { background: #2e0505; border: 2px solid #dc2626; color: #fca5a5; }
    .hold  { background: #1a1a1a; border: 2px solid #6b7280; color: #d1d5db; }
    .big-signal   { font-size: 52px; font-weight: 800; letter-spacing: 2px; }
    .big-caption  { font-size: 14px; opacity: 0.85; margin-top: 6px; }
    .stat-row { display: flex; justify-content: space-around; margin-top: 14px; }
    .stat-val { font-size: 20px; font-weight: 700; }
    .stat-lbl { font-size: 11px; opacity: 0.7; text-transform: uppercase; letter-spacing: 1px; }
    .price-up   { color: #4ade80; }
    .price-down { color: #f87171; }
    .gauge-wrap { margin: 14px 30px 4px 30px; }
    .gauge-bar {
        position: relative; height: 10px; border-radius: 5px;
        background: linear-gradient(90deg, #16a34a, #84cc16, #eab308, #f97316, #dc2626);
    }
    .gauge-dot {
        position: absolute; top: -5px; width: 6px; height: 20px;
        background: #fff; border-radius: 3px; box-shadow: 0 0 6px rgba(0,0,0,.6);
    }
    .gauge-labels {
        display: flex; justify-content: space-between;
        font-size: 10px; opacity: 0.6; margin-top: 6px;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Long or Short?")
st.caption("US + Hong Kong stocks · ML signal + Wall Street consensus")


# ─── Symbol handling (HK-aware) ───
def normalize_symbol(raw: str) -> str:
    """'700' → '0700.HK' · '700.hk' → '0700.HK' · 'aapl' → 'AAPL'."""
    s = raw.strip().upper()
    if not s:
        return s
    if s.isdigit():  # bare number = Hong Kong convention
        return s.zfill(4) + ".HK"
    if s.endswith(".HK"):
        num = s.split(".")[0]
        if num.isdigit():
            return num.zfill(4) + ".HK"
    return s


def currency_prefix(sym: str, analyst: dict | None) -> str:
    cur = (analyst or {}).get("currency")
    if cur == "HKD" or sym.endswith(".HK"):
        return "HK$"
    if cur in (None, "USD"):
        return "$"
    return f"{cur} "


# ─── One-tap popular tickers (Google Finance pattern) ───
POPULAR = [
    ("SPY", "S&P 500"), ("AAPL", "Apple"), ("NVDA", "Nvidia"), ("TSLA", "Tesla"),
    ("0700.HK", "Tencent"), ("9988.HK", "Alibaba"), ("1299.HK", "AIA"),
    ("0005.HK", "HSBC"), ("3690.HK", "Meituan"), ("BTC-USD", "Bitcoin"),
]


def _pick(sym: str) -> None:
    st.session_state.sym_input = sym


st.text_input(
    "Search any US or HK stock",
    key="sym_input",
    placeholder="AAPL · NVDA · 0700.HK · or just 700 for Tencent",
)

chip_rows = [POPULAR[:5], POPULAR[5:]]
for row in chip_rows:
    cols = st.columns(len(row))
    for col, (sym, name) in zip(cols, row):
        col.button(name, key=f"chip_{sym}", on_click=_pick, args=(sym,), use_container_width=True)

symbol = normalize_symbol(st.session_state.get("sym_input", ""))
if not symbol:
    st.info("Pick a stock above or type a symbol — results appear instantly.")
    st.stop()

MIN_CONFIDENCE = 0.25

# Proven-edge gate: BUY/SELL only fires if the walk-forward backtest on THIS
# symbol historically cleared these bars.
MIN_EDGE_WINRATE = 0.55
MIN_EDGE_TRADES = 15
MIN_EDGE_SHARPE = 0.3

# 4 years: ~1000 bars → 500 to train walk-forward + ~500 out-of-sample to verify.
START_ISO = (date.today() - timedelta(days=4 * 365)).isoformat()


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
        "name": view.name,
        "currency": view.currency,
        "rec_mean": view.rec_mean,
    }


@st.cache_data(show_spinner=False)
def get_ml_signal(sym: str, _df_hash: int) -> dict:
    """Ensemble ML signal at multiple horizons; return the strongest signal."""
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
    return max(signals, key=lambda s: s.get("confidence", 0.0))


@st.cache_data(show_spinner=False)
def check_edge(sym: str, _df_hash: int) -> dict | None:
    """Walk-forward backtest on this exact symbol: does the model actually
    have a historical edge here? None = can't verify = don't trade."""
    try:
        macro_df = load_macro()
        wf = ml.EnsembleWalkForward(train_size=500, step_size=60)
        positions = wf.positions(df, macro_df=macro_df)
        result = backtest.run(df, positions, fee_bps=1.0, slippage_bps=1.0)
        m = result.metrics
        return {
            "win_rate": float(m.get("win_rate", 0.0)),
            "sharpe": float(m.get("sharpe", 0.0)),
            "trades": int(m.get("trades", 0)),
            "total_return": float(m.get("total_return", 0.0)),
        }
    except Exception:
        return None


def compute_trend(df: pd.DataFrame) -> tuple[str, str]:
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


def gated_signal(
    ml_sig: dict,
    analyst: dict | None,
    edge: dict | None,
    trend_label: str,
) -> tuple[str, str, str]:
    """All gates: confidence, proven edge, trend veto, analyst consensus.
    Any gate failure → HOLD, because an unverified signal is how money gets lost."""
    ml_dir = ml_sig.get("signal", "flat")
    conf = ml_sig.get("confidence", 0.0)

    if ml_dir == "flat":
        return "HOLD", "hold", "Model has no directional edge right now"

    if conf < MIN_CONFIDENCE:
        return "HOLD", "hold", f"Model confidence {conf*100:.0f}% below {MIN_CONFIDENCE*100:.0f}% threshold — not worth the risk"

    if edge is None:
        return "HOLD", "hold", "Not enough history to verify the model works on this symbol — refusing to guess"
    if edge["trades"] < MIN_EDGE_TRADES:
        return "HOLD", "hold", f"Only {edge['trades']} historical trades on this symbol — sample too small to trust"
    if edge["win_rate"] < MIN_EDGE_WINRATE or edge["sharpe"] < MIN_EDGE_SHARPE:
        return "HOLD", "hold", (
            f"Model's track record here is weak ({edge['win_rate']*100:.0f}% win rate, "
            f"Sharpe {edge['sharpe']:.2f}) — it has no proven edge on {symbol}"
        )

    if ml_dir == "long" and trend_label in ("STRONG DOWNTREND", "VOLATILE / CRASHING"):
        return "HOLD", "hold", "Model bullish but price is in a strong downtrend — not catching falling knives"
    if ml_dir == "short" and trend_label == "STRONG UPTREND":
        return "HOLD", "hold", "Model bearish but price is in a strong uptrend — not fighting momentum"

    if analyst is not None and analyst.get("num_analysts", 0) >= 5:
        an_dir = analyst.get("direction", "flat")
        if ml_dir == "long" and an_dir == "down":
            return "HOLD", "hold", "Model bullish but analysts see downside — no consensus"
        if ml_dir == "short" and an_dir == "up":
            return "HOLD", "hold", "Model bearish but analysts see upside — no consensus"

    track = f"verified {edge['win_rate']*100:.0f}% historical win rate on {symbol} ({edge['trades']} trades)"
    if ml_dir == "long":
        rationale = f"Model {conf*100:.0f}% confident of upside · {track}"
        if analyst and analyst.get("num_analysts", 0) >= 5 and analyst.get("direction") == "up":
            rationale += f" · {analyst['num_analysts']} analysts agree"
        return "BUY", "buy", rationale
    rationale = f"Model {conf*100:.0f}% confident of downside · {track}"
    if analyst and analyst.get("num_analysts", 0) >= 5 and analyst.get("direction") == "down":
        rationale += f" · {analyst['num_analysts']} analysts agree"
    return "SELL", "sell", rationale


# ─── Load ───
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

with st.spinner(f"Verifying the model's track record on {symbol} (walk-forward backtest)..."):
    edge = check_edge(symbol, len(df))

# ─── Compute ───
trend_label, trend_emoji = compute_trend(df)
signal, css, rationale = gated_signal(ml_sig, analyst, edge, trend_label)
current_price = float(df["close"].iloc[-1])
prev_price = float(df["close"].iloc[-2])
day_chg = (current_price / prev_price - 1) * 100
cur = currency_prefix(symbol, analyst)
display_name = (analyst or {}).get("name") or symbol

# ─── Header: name, price, day change (Robinhood pattern) ───
chg_cls = "price-up" if day_chg >= 0 else "price-down"
st.markdown(
    f"""
    <h3 style="margin-bottom:0;">{display_name} <span style="opacity:.6; font-size:.7em;">{symbol}</span></h3>
    <div style="font-size:34px; font-weight:800;">
        {cur}{current_price:,.2f}
        <span class="{chg_cls}" style="font-size:20px;">{day_chg:+.2f}% today</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Price chart (1y) with analyst target line ───
hist = df[["close"]].tail(252).reset_index()
hist.columns = ["date", "close"]
line_color = "#4ade80" if hist["close"].iloc[-1] >= hist["close"].iloc[0] else "#f87171"
chart = alt.Chart(hist).mark_line(color=line_color, strokeWidth=2).encode(
    x=alt.X("date:T", axis=alt.Axis(title=None, grid=False)),
    y=alt.Y("close:Q", scale=alt.Scale(zero=False), axis=alt.Axis(title=None)),
)
if analyst and analyst.get("mean_target"):
    target_rule = alt.Chart(pd.DataFrame({"y": [analyst["mean_target"]]})).mark_rule(
        strokeDash=[6, 4], color="#fbbf24", strokeWidth=2
    ).encode(y="y:Q")
    chart = chart + target_rule
st.altair_chart(chart.properties(height=220), use_container_width=True)
if analyst and analyst.get("mean_target"):
    st.caption(f"— price (1y)  · · · analyst consensus target {cur}{analyst['mean_target']:,.2f}")

# ─── 1. Trend ───
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

# ─── 2. Buy / Sell / Hold with exit plan ───
atr = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else current_price * 0.02
if signal == "BUY":
    stop_price = current_price - 2.5 * atr
    tp_price = current_price + 2.5 * atr
    exit_plan = (
        f"Stop loss: {cur}{stop_price:,.2f} (−{2.5*atr/current_price*100:.1f}%) · "
        f"Take profit: {cur}{tp_price:,.2f} · Exit after 30 trading days regardless"
    )
elif signal == "SELL":
    stop_price = current_price + 2.5 * atr
    tp_price = current_price - 2.5 * atr
    exit_plan = (
        f"Stop loss: {cur}{stop_price:,.2f} (+{2.5*atr/current_price*100:.1f}%) · "
        f"Take profit: {cur}{tp_price:,.2f} · Exit after 30 trading days regardless"
    )
else:
    exit_plan = "Stay out. Cash is a position — a skipped bad trade is money kept."

st.markdown(
    f"""
    <div class="big-card {css}">
      <div class="stat-lbl">Recommendation</div>
      <div class="big-signal">{signal}</div>
      <div class="big-caption">{rationale}</div>
      <div class="big-caption" style="margin-top:12px; font-weight:600;">{exit_plan}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if edge is not None:
    edge_ok = edge["win_rate"] >= MIN_EDGE_WINRATE and edge["sharpe"] >= MIN_EDGE_SHARPE and edge["trades"] >= MIN_EDGE_TRADES
    verdict = "✓ verified edge" if edge_ok else "✗ no proven edge — signals blocked"
    st.caption(
        f"Model track record on {symbol} (walk-forward, ~2y out-of-sample): "
        f"{edge['win_rate']*100:.0f}% win rate · Sharpe {edge['sharpe']:.2f} · "
        f"{edge['trades']} trades · {edge['total_return']*100:+.1f}% return · {verdict}"
    )

# ─── 3. Wall Street target with Yahoo-style rating gauge ───
if analyst and analyst.get("mean_target"):
    mean = analyst["mean_target"]
    high = analyst.get("high_target")
    low = analyst.get("low_target")
    n = analyst["num_analysts"]
    upside = analyst.get("upside_pct")
    upside_str = f"{upside:+.1f}%" if upside is not None else "?"
    conviction = "high conviction" if analyst["is_high_conviction"] else f"{n} analyst{'s' if n != 1 else ''}"

    # Yahoo-style gauge: recommendationMean is 1.0 (strong buy) → 5.0 (strong sell)
    gauge_html = ""
    rec_mean = analyst.get("rec_mean")
    if rec_mean is not None:
        pct = max(0.0, min(100.0, (rec_mean - 1.0) / 4.0 * 100.0))
        gauge_html = f"""
          <div class="gauge-wrap">
            <div class="gauge-bar"><div class="gauge-dot" style="left:{pct:.0f}%;"></div></div>
            <div class="gauge-labels">
              <span>Strong Buy</span><span>Buy</span><span>Hold</span><span>Sell</span><span>Strong Sell</span>
            </div>
          </div>
        """

    st.markdown(
        f"""
        <div class="big-card hold">
          <div class="stat-lbl">Wall Street Target (12-month)</div>
          <div class="big-signal">{cur}{mean:,.2f}</div>
          <div class="big-caption">{upside_str} from {cur}{current_price:,.2f} · {conviction}</div>
          {gauge_html}
          <div class="stat-row">
            <div>
              <div class="stat-lbl">Low</div>
              <div class="stat-val">{cur}{low:,.2f}</div>
            </div>
            <div>
              <div class="stat-lbl">Consensus</div>
              <div class="stat-val">{cur}{mean:,.2f}</div>
            </div>
            <div>
              <div class="stat-lbl">High</div>
              <div class="stat-val">{cur}{high:,.2f}</div>
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
          <div class="big-caption">No analyst coverage for this symbol (common for crypto / ETFs / indexes)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "BUY/SELL only fires when ALL gates pass: model confidence ≥25%, verified historical "
    "edge on this exact symbol (≥55% win rate, positive Sharpe), trend alignment, and "
    "analyst-consensus agreement. Risk max 1–2% of your account per trade and always place "
    "the stop-loss order. Not financial advice. Past performance does not guarantee future results."
)
