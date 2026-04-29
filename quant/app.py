"""Streamlit app: enter a symbol, get a long/short/flat call right now.

Recommendation = majority vote of three strategies (regime breakout, RSI
mean-reversion, MA crossover) computed on the latest bar. The ensemble's
own historical win ratio on the same symbol is shown as context.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

# Sklearn complains when fit-with-DataFrame and predict-with-array have feature
# name mismatches. Cosmetic — doesn't affect predictions. Suppress in the app.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from data import sources
from data import live as live_quote_mod
import indicators
import regime
import backtest
import strategies
import ml
import class_xgb
import options
import social_scanner
import screener

st.set_page_config(page_title="Long or Short?", page_icon="$", layout="centered")

st.title("Long or Short?")
st.caption("XGBoost predicts next-10-day direction · Hold up to 30 days · Stop at 2.5×ATR · ONLY recommends when confidence > 75% (most queries return FLAT — by design)")

# Model-speed selection lives at the top so it applies to BOTH the screener
# (which scans 25 symbols) and the per-symbol analysis below.
model_speed = st.radio(
    "Model",
    options=["Fast (XGB)", "Best (5-model ensemble)"],
    index=1,
    help="Fast: 1 model, ~2s. Best: avg of 5 models, ~10s, +0.07 Sharpe.",
    horizontal=True,
)
use_ensemble = model_speed.startswith("Best")

with st.expander("How this strategy works", expanded=False):
    st.markdown(
        """
**The model**
- XGBoost classifier predicts P(price up over next 10 trading days) from technicals + regime + lagged returns
- Class-pooled training when symbol is equity / broad-ETF / crypto (more data, better generalization)
- Tuned hyperparameters: horizon=10, max_depth=3, n_estimators=100 (selected by FDR-corrected grid search)

**When to act (current threshold: 75% confidence)**
- **BUY LONG** if P(up) > **0.875**
- **SELL SHORT** if P(up) < **0.125**
- **STAY FLAT** otherwise — almost everything at this threshold

⚠ **Important nuance.** "Confidence" here = `|P(up) − 0.5| × 2`. It's distance
from coin-flip, NOT a calibrated probability of being right. At 75% confidence
(P(up) > 0.875), backtests show empirical win rate of ~60-65%, not 75%. The
75% filter mainly serves to *only fire on the strongest possible signals* —
expect to see FLAT on most queries. If you want more recommendations,
lower MIN_CONFIDENCE in app.py.

**How to size the trade**
- Stop loss: **2.5× ATR(14)** against you (volatility-adjusted, tighter for SPY, wider for crypto)
- Take profit: **2.5× ATR(14)** in your favor (1:1 reward:risk)
- Time stop: exit after **~30 trading days** (~6 weeks) regardless — model's edge decays past that

**Validated on 12 months × 10 symbols (Setup B)**
- 56.5% win rate · +16.9% portfolio return · Sharpe 0.97 · Max DD -16.1%
- 186 trades — fewer than the original 10-bar time stop (265) because longer holds let trades reach take-profit instead of timing out

**Key honest caveat**
- Strategy underperforms buy-and-hold in strong bull markets (gives up alpha to maintain hedging)
- Outperforms in bear/crashing markets (e.g., BTC backtest: +23% strategy vs -17% B&H)
- ~44% of trades will be losers — only manageable with strict ATR stops + position sizing
- Past backtest performance does not guarantee future results

**Position sizing rule of thumb**
- Risk **1-2% of account per trade** (use the "risk $/share" number under the recommendation)
- Example: $10k account, risk $100/trade, AAPL stop $4/share → 25 shares
        """
    )

# ─── Trending tickers (apewisdom — Reddit + StockTwits + Twitter aggregator) ───
with st.expander("Trending tickers", expanded=False):
    st.caption("Live cross-platform scan via apewisdom.io (aggregates Reddit + StockTwits + Twitter). No auth needed. Ranked by mentions × 24h momentum.")

    @st.cache_data(show_spinner=False, ttl=300)  # 5-min cache
    def cached_social_scan(filter_name: str, _ts: int) -> list[dict]:
        results = social_scanner.scan(limit=25, filter_name=filter_name)
        return [{"ticker": r.ticker, "company": r.company_name,
                 "mentions": r.mentions, "yesterday": r.mentions_24h_ago,
                 "momentum": r.momentum, "rank_chg": r.rank_change,
                 "score": r.score} for r in results]

    sc1, sc2 = st.columns([2, 2])
    filt = sc1.selectbox(
        "Source",
        options=list(social_scanner.VALID_FILTERS),
        index=0,
        help="all-stocks: combined feed. wallstreetbets/options/etc: source-specific. cryptos: BTC/ETH/etc.",
    )
    if sc2.button("Scan now", use_container_width=True):
        with st.spinner(f"Scanning apewisdom ({filt})..."):
            try:
                import time as _t
                rows = cached_social_scan(filt, int(_t.time() // 300))
                if not rows:
                    st.info("No tickers returned.")
                else:
                    df_st = pd.DataFrame(rows)
                    st.dataframe(
                        df_st[["ticker", "company", "mentions", "yesterday", "momentum", "rank_chg", "score"]],
                        use_container_width=True, hide_index=True,
                        column_config={
                            "ticker": st.column_config.TextColumn("Ticker", width="small"),
                            "company": st.column_config.TextColumn("Company", width="medium"),
                            "mentions": st.column_config.NumberColumn("Today", width="small", help="Mentions in last ~24h"),
                            "yesterday": st.column_config.NumberColumn("24h ago", width="small"),
                            "momentum": st.column_config.NumberColumn("Mom×", format="%.1fx", width="small", help="today / yesterday — >1.0 = heating up"),
                            "rank_chg": st.column_config.NumberColumn("Rank Δ", format="%+d", width="small", help="Positive = climbed in popularity"),
                            "score": st.column_config.NumberColumn("Score", format="%.1f", width="small"),
                        },
                    )
                    top3 = ", ".join(r["ticker"] for r in rows[:3])
                    st.caption(f"Drop one into the Symbol box below. Top 3: **{top3}**")
            except Exception as e:
                st.error(f"Scan failed: {e}")

# ─── Stock screener (find actionable signals across the universe) ───
with st.expander("Stock screener — find high-conviction signals", expanded=False):
    st.caption(f"Scans 25 liquid symbols (mega-caps + ETFs + crypto). Surfaces only those meeting the confidence threshold (currently {0.75:.0%}). Most days: 0 hits — by design.")

    @st.cache_data(show_spinner=False, ttl=2 * 3600)
    def cached_screen(ensemble_flag: bool, _bucket: int) -> list[dict]:
        rows = screener.screen(use_ensemble=ensemble_flag)
        return [{
            "ticker": r.ticker, "class": r.asset_class,
            "signal": r.signal, "proba": r.proba,
            "confidence": r.confidence, "last_close": r.last_close,
        } for r in rows]

    sc_col1, sc_col2 = st.columns([3, 1])
    sc_col1.write("**Click to scan** — first run trains models for ~25 symbols (~30-90s); cached for 2h afterward.")
    if sc_col2.button("Run screen", use_container_width=True, key="run_screen"):
        st.session_state["screen_run"] = True

    if st.session_state.get("screen_run"):
        with st.spinner("Scanning 25 symbols..."):
            try:
                import time as _t
                screen_rows = cached_screen(use_ensemble, int(_t.time() // (2 * 3600)))
            except Exception as e:
                st.error(f"Screen failed: {e}")
                screen_rows = []

        if screen_rows:
            actionable = [r for r in screen_rows
                          if r["confidence"] >= 0.75 and r["signal"] != "flat"]

            if actionable:
                st.markdown(f"### Actionable signals ({len(actionable)})")
                df_act = pd.DataFrame([{
                    "Ticker": r["ticker"],
                    "Signal": r["signal"].upper(),
                    "P(up)": r["proba"],
                    "Conf": r["confidence"],
                    "Price": r["last_close"],
                    "Class": r["class"],
                } for r in actionable])
                st.dataframe(
                    df_act, use_container_width=True, hide_index=True,
                    column_config={
                        "P(up)": st.column_config.NumberColumn(format="%.1f%%"),
                        "Conf": st.column_config.NumberColumn(format="%.0f%%"),
                        "Price": st.column_config.NumberColumn(format="$%.2f"),
                    },
                )
                top_tickers = ", ".join(r["ticker"] for r in actionable[:5])
                st.caption(f"Drop one into the Symbol input below for full analysis. Top: **{top_tickers}**")
            else:
                st.info("No symbol hit the 75% confidence threshold. The model is leaning weakly across all 25. Either wait for stronger market signals, or lower MIN_CONFIDENCE in app.py to see all leans.")

            with st.expander(f"All {len(screen_rows)} results (sorted by confidence)"):
                df_all = pd.DataFrame([{
                    "Ticker": r["ticker"],
                    "Class": r["class"],
                    "Lean": r["signal"].upper(),
                    "P(up)": r["proba"],
                    "Conf": r["confidence"],
                    "Price": r["last_close"],
                    "Actionable": "yes" if (r["confidence"] >= 0.75 and r["signal"] != "flat") else "—",
                } for r in screen_rows])
                st.dataframe(
                    df_all, use_container_width=True, hide_index=True,
                    column_config={
                        "P(up)": st.column_config.NumberColumn(format="%.1f%%"),
                        "Conf": st.column_config.NumberColumn(format="%.0f%%"),
                        "Price": st.column_config.NumberColumn(format="$%.2f"),
                    },
                )

symbol = st.text_input(
    "Symbol", value="SPY",
    help="e.g. AAPL, SPY, QQQ, BTC-USD",
).upper().strip()

# Cache lifetime for data + model predictions. After this, fresh OHLC fetched,
# models retrained. Live spot quote refreshes separately at 30s.
CACHE_TTL_SECONDS = 2 * 3600  # 2 hours
AUTO_REFRESH_MS = CACHE_TTL_SECONDS * 1000  # full page rerun cadence

# Minimum confidence to recommend a trade. 0.75 confidence == |P(up) - 0.5| > 0.375,
# so effective long/short thresholds are P(up) > 0.875 or < 0.125. Very high bar —
# most signals will resolve to FLAT. Honest note: model "confidence" is distance
# from coin-flip, NOT calibrated probability of being right. Empirical win rate
# at these extreme probabilities is ~60-65% from backtests, not 75%.
MIN_CONFIDENCE = 0.75


def apply_confidence_filter(sig: dict, threshold: float = MIN_CONFIDENCE) -> dict:
    """Override signal to 'flat' if confidence below threshold. Records original."""
    if not sig or "confidence" not in sig:
        return sig
    if sig["confidence"] < threshold and sig.get("signal") != "flat":
        out = dict(sig)
        out["signal_raw"] = sig["signal"]
        out["signal"] = "flat"
        out["filtered"] = True
        return out
    return sig

clicked = st.button("Get recommendation", type="primary", use_container_width=True)

# Persist the user's request across auto-refresh re-runs. After 2h, page rerun
# fires WITHOUT a button click — but if the same symbol+model are still active,
# we re-render with the fresh cache instead of dropping back to the empty form.
if clicked:
    st.session_state["active_symbol"] = symbol
    st.session_state["active_use_ensemble"] = use_ensemble

active_symbol = st.session_state.get("active_symbol")
active_use_ensemble = st.session_state.get("active_use_ensemble")
should_render = clicked or (
    active_symbol == symbol and active_use_ensemble == use_ensemble
)

if not should_render:
    st.stop()

if not symbol:
    st.error("Enter a stock symbol.")
    st.stop()

# Once we're rendering an active analysis, schedule auto-refresh so the page
# re-runs and picks up the freshly-expired caches.
st_autorefresh(interval=AUTO_REFRESH_MS, key="auto_refresh_2h")


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def load(sym: str) -> pd.DataFrame:
    start = (date.today() - timedelta(days=3 * 365)).isoformat()
    df = sources.get_bars(sym, start, use_cache=True)
    df = indicators.attach_all(df)
    df = regime.label(df)
    return df


with st.spinner(f"Analyzing {symbol}..."):
    try:
        df = load(symbol)
    except Exception as e:
        st.error(f"Could not load **{symbol}**: {e}")
        st.stop()

if df.empty or len(df) < 60:
    st.error(f"Not enough data for {symbol}.")
    st.stop()

# ─── Determine model approach: class-pooled or per-symbol ───
sym_class = class_xgb.classify_symbol(symbol)
use_pooled = sym_class in class_xgb.POOL_BENEFITS
start_date_iso = (date.today() - timedelta(days=3 * 365)).isoformat()


# ─── Cached model functions ───
# Cache key includes `ensemble_flag` so toggling the radio invalidates correctly.

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def cached_pooled_today(class_name: str, start_iso: str, ensemble_flag: bool) -> dict:
    if ensemble_flag:
        return class_xgb.predict_today_for_class_ensemble(class_name, start_iso)
    return class_xgb.predict_today_for_class(class_name, start_iso)


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def cached_pooled_walkforward(class_name: str, start_iso: str, ensemble_flag: bool) -> dict:
    if ensemble_flag:
        return class_xgb.predict_class_ensemble(
            class_name, start_iso, horizon=10, train_size=750, step_size=60
        )
    return class_xgb.predict_class(
        class_name, start_iso, horizon=10, train_size=750, step_size=60
    )


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def cached_per_symbol_today(sym: str, _df_hash: int, ensemble_flag: bool) -> dict:
    if ensemble_flag:
        predictor = ml.EnsemblePredictor().fit(df)
    else:
        predictor = ml.XGBPredictor().fit(df)
    sig = predictor.predict_now(df)
    sig["importance"] = predictor.feature_importance().head(8).to_dict()
    return sig


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def cached_per_symbol_walkforward(sym: str, _df_hash: int, ensemble_flag: bool) -> dict | None:
    try:
        if ensemble_flag:
            wf = ml.EnsembleWalkForward()
            positions = wf.positions(df)
        else:
            positions = strategies.xgb_walk_forward(df)
        return backtest.run(df, positions, fee_bps=1.0, slippage_bps=1.0).metrics
    except Exception:
        return None


# ─── Multi-horizon prediction helpers ───
@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def cached_pooled_today_horizon(class_name: str, start_iso: str, ensemble_flag: bool, horizon: int) -> dict:
    if ensemble_flag:
        return class_xgb.predict_today_for_class_ensemble(class_name, start_iso, horizon=horizon)
    return class_xgb.predict_today_for_class(class_name, start_iso, horizon=horizon)


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def cached_per_symbol_today_horizon(sym: str, _df_hash: int, ensemble_flag: bool, horizon: int) -> dict:
    if ensemble_flag:
        predictor = ml.EnsemblePredictor(horizon=horizon).fit(df)
    else:
        predictor = ml.XGBPredictor(horizon=horizon).fit(df)
    return predictor.predict_now(df)


# ─── Today's signal (pooled if class benefits, else per-symbol) ───
model_label = ""
ensemble_suffix = " · 5-model ensemble" if use_ensemble else " · XGBoost"

if use_pooled:
    n_class_symbols = len(class_xgb.CLASSES[sym_class])
    kind = "5-model ensemble" if use_ensemble else "XGBoost"
    spinner_msg = (
        f"Training pooled {kind} on {sym_class} class "
        f"({n_class_symbols} symbols)..."
    )
    with st.spinner(spinner_msg):
        try:
            pool_results = cached_pooled_today(sym_class, start_date_iso, use_ensemble)
            xgb_sig = pool_results.get(symbol)
            if xgb_sig is None:
                raise RuntimeError(f"{symbol} not in pooled predictions")
            n_models = xgb_sig.get("n_models", 1)
            model_label = (
                f"class-pooled · {sym_class} ({xgb_sig['n_symbols_pooled']} symbols, "
                f"{xgb_sig['n_training_rows']:,} pooled training rows){ensemble_suffix}"
                + (f" — averaging {n_models} models" if use_ensemble else "")
            )
        except Exception as e:
            st.warning(f"Pooled training failed ({e}); falling back to per-symbol path.")
            use_pooled = False

if not use_pooled:
    kind = "5-model ensemble" if use_ensemble else "XGBoost"
    msg = (
        f"Training per-symbol {kind}..."
        if sym_class is None
        else f"Training per-symbol {kind} ({sym_class} class — pooling not beneficial)..."
    )
    with st.spinner(msg):
        try:
            xgb_sig = cached_per_symbol_today(symbol, len(df), use_ensemble)
            base = (
                "per-symbol (no asset-class match)"
                if sym_class is None
                else f"per-symbol (class={sym_class}, pooling not beneficial)"
            )
            model_label = base + ensemble_suffix
        except Exception as e:
            st.error(f"ML training failed: {e}")
            st.stop()

# Apply confidence filter — overrides to FLAT when conf < 15%.
xgb_sig = apply_confidence_filter(xgb_sig)
xgb_call = {"long": 1, "short": -1, "flat": 0}[xgb_sig["signal"]]

# ─── Rule-based strategies (secondary context) ───
positions_by_strategy = {
    "regime_breakout": strategies.regime_aware_breakout(df),
    "rsi_meanrev": strategies.rsi_mean_reversion(df),
    "ma_cross": strategies.ma_crossover(df),
}
current_calls = {name: int(p.iloc[-1]) for name, p in positions_by_strategy.items()}

# ─── Rule-based ensemble metrics (kept for the fallback branch only) ───
ensemble_series = (
    sum(positions_by_strategy.values())
    .apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
)
m = backtest.run(df, ensemble_series, fee_bps=1.0, slippage_bps=1.0).metrics

# ─── XGB historical win rate on this symbol (walk-forward) ───
xgb_metrics = None
if use_pooled:
    kind = "5-model ensemble" if use_ensemble else "XGBoost"
    with st.spinner(f"Walk-forward backtesting pooled {kind} on {sym_class}..."):
        try:
            probas_by_sym = cached_pooled_walkforward(sym_class, start_date_iso, use_ensemble)
            proba = probas_by_sym.get(symbol)
            if proba is not None and not proba.dropna().empty:
                proba_aligned = proba.reindex(df.index)
                positions = pd.Series(0.0, index=df.index)
                positions[proba_aligned > 0.55] = 1.0
                positions[proba_aligned < 0.45] = -1.0
                xgb_metrics = backtest.run(df, positions, fee_bps=1.0, slippage_bps=1.0).metrics
        except Exception:
            xgb_metrics = None
else:
    kind = "5-model ensemble" if use_ensemble else "XGBoost"
    with st.spinner(f"Walk-forward backtesting per-symbol {kind}..."):
        xgb_metrics = cached_per_symbol_walkforward(symbol, len(df), use_ensemble)

# XGB sole signal — rule votes are display-only.
recommendation_call = xgb_call

# ─── Header ───
last_close = df["close"].iloc[-1]
prev_close = df["close"].iloc[-2]
last_date = df.index[-1].date()
day_change_pct = (last_close / prev_close - 1) * 100
current_regime = df["regime"].iloc[-1] or "unclassified"
current_rsi = df["rsi_14"].iloc[-1]
current_adx = df["adx"].iloc[-1]
sma20 = df["close"].rolling(20).mean().iloc[-1]
sma50 = df["close"].rolling(50).mean().iloc[-1]
src_name = df.attrs.get("source", "?")

st.markdown(
    f"### {symbol} &nbsp;·&nbsp; ${last_close:,.2f} &nbsp;·&nbsp; "
    f"{day_change_pct:+.2f}% &nbsp;·&nbsp; as of {last_date}",
    unsafe_allow_html=True,
)

# ─── Live spot quote ───
@st.cache_data(show_spinner=False, ttl=30)
def cached_live_quote(sym: str, _bucket: int) -> dict | None:
    return live_quote_mod.get_live_quote(sym)

import time as _time_mod
live = cached_live_quote(symbol, int(_time_mod.time() // 30))

if live and live["price"] > 0:
    gap_pct = (live["price"] / last_close - 1) * 100
    is_stale = abs(gap_pct) > 1.0
    live_color = "#00E68A" if live["change_pct"] >= 0 else "#FF4757"
    fetched_str = _time_mod.strftime("%H:%M:%S", _time_mod.localtime(live["timestamp"]))
    stale_note = (
        f"<span style='color:#FFD93D; margin-left:8px;'>price moved {gap_pct:+.1f}% since model input — signal may be stale</span>"
        if is_stale else ""
    )
    st.markdown(
        f"""
        <div style="background:#1A1B24; border:1px solid #2A2B38; border-radius:8px;
                    padding:10px 14px; margin:0 0 12px 0; font-size:12px;">
            <span style="font-size:10px; color:#888; letter-spacing:1px;">LIVE</span>
            &nbsp;<b style="font-size:18px; color:#fff;">${live['price']:,.2f}</b>
            <span style="color:{live_color}; font-weight:600; margin-left:6px;">
                {live['change']:+.2f} ({live['change_pct']:+.2f}%)
            </span>
            <span style="color:#666; margin-left:6px;">today · range ${live['day_low']:,.2f}–${live['day_high']:,.2f}</span>
            <span style="color:#666; margin-left:8px; font-size:10px;">refreshed {fetched_str}</span>
            {stale_note}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Big recommendation card (XGBoost is the headline) ───
action_label = {1: "BUY LONG", -1: "SELL SHORT", 0: "STAY FLAT"}[recommendation_call]
action_color = {1: "#00E68A", -1: "#FF4757", 0: "#FF9F43"}[recommendation_call]

filtered_note = ""
if xgb_sig.get("filtered"):
    raw_sig = xgb_sig.get("signal_raw", "?").upper()
    filtered_note = f"<div style='font-size:10px; color:#FFD93D; margin-top:4px;'>filtered: model leaned {raw_sig} but conf {xgb_sig['confidence']:.0%} below 15% threshold</div>"

st.markdown(
    f"""
    <div style="background:{action_color}22; border:2px solid {action_color};
                border-radius:12px; padding:24px; text-align:center; margin:16px 0;">
        <div style="font-size:11px; color:#888; letter-spacing:1.5px;">10-DAY PREDICTION</div>
        <div style="font-size:42px; font-weight:700; color:{action_color}; margin:8px 0;">{action_label}</div>
        <div style="font-size:13px; color:#aaa;">
            P(up) <b>{xgb_sig['proba']:.1%}</b> · conf <b>{xgb_sig['confidence']:.0%}</b>
        </div>
        {filtered_note}
        <div style="font-size:10px; color:#666; margin-top:6px;">{model_label}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── ATR-based stop suggestion ───
if recommendation_call != 0:
    atr_value = df["atr_14"].iloc[-1]
    if not pd.isna(atr_value) and last_close > 0:
        atr_pct = float(atr_value) / float(last_close)
        stop_dist_pct = 2.5 * atr_pct
        is_long = recommendation_call > 0
        if is_long:
            stop_price = last_close * (1 - stop_dist_pct)
            tp_price = last_close * (1 + stop_dist_pct)
            stop_dir, tp_dir = "below", "above"
        else:
            stop_price = last_close * (1 + stop_dist_pct)
            tp_price = last_close * (1 - stop_dist_pct)
            stop_dir, tp_dir = "above", "below"
        risk_per_share = abs(last_close - stop_price)
        # Setup B: 30 trading-day time stop (~42 calendar days). The model predicts
        # the next 10-day direction, but holding longer captures more of the move
        # — TP exits jumped from 29% → 49% with the longer time stop in backtests.
        time_stop = (last_date + timedelta(days=42)).strftime("%b %d")
        st.markdown(
            f"""
            <div style="background:#1A1B24; border:1px solid #2A2B38; border-left:3px solid {action_color};
                        border-radius:6px; padding:10px 14px; margin:-10px 0 16px 0; font-size:12px; color:#aaa;">
                <b style="color:#ccc;">Suggested levels</b> <span style="color:#666;">(2.5× ATR, 1:1 R:R)</span>
                &nbsp;·&nbsp; Entry <b style="color:#fff;">${last_close:.2f}</b>
                &nbsp;·&nbsp; Stop <b style="color:#FF4757;">${stop_price:.2f}</b>
                <span style="color:#666;">({stop_dist_pct*100:.1f}% {stop_dir}, risk ${risk_per_share:.2f}/sh)</span>
                &nbsp;·&nbsp; TP <b style="color:#00E68A;">${tp_price:.2f}</b>
                <span style="color:#666;">({stop_dist_pct*100:.1f}% {tp_dir})</span>
                &nbsp;·&nbsp; Time-stop <b style="color:#FFD93D;">{time_stop}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ─── Longer-horizon predictions (30d + 60d) ───
st.subheader("Longer horizons")
horizon_sigs: dict[int, dict | None] = {}
for h in (30, 60):
    try:
        if use_pooled:
            with st.spinner(f"Training {h}-day model..."):
                pool_h = cached_pooled_today_horizon(sym_class, start_date_iso, use_ensemble, h)
                raw = pool_h.get(symbol)
        else:
            with st.spinner(f"Training {h}-day model..."):
                raw = cached_per_symbol_today_horizon(symbol, len(df), use_ensemble, h)
        horizon_sigs[h] = apply_confidence_filter(raw) if raw else None
    except Exception as e:
        horizon_sigs[h] = {"error": str(e)}

hcols = st.columns(2)
for col, h in zip(hcols, (30, 60)):
    sig = horizon_sigs.get(h)
    if sig is None or "error" in (sig or {}):
        err = (sig or {}).get("error", "no result")
        col.markdown(
            f"""
            <div style="background:#1A1B24; border:1px solid #2A2B38; border-radius:8px; padding:14px;">
                <div style="font-size:11px; color:#888; letter-spacing:1px;">{h}-DAY PREDICTION</div>
                <div style="font-size:13px; color:#FF9F43; margin-top:8px;">unavailable</div>
                <div style="font-size:10px; color:#666; margin-top:4px;">{err[:80]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        continue
    h_call = {"long": 1, "short": -1, "flat": 0}[sig["signal"]]
    h_label = {1: "BUY LONG", -1: "SELL SHORT", 0: "STAY FLAT"}[h_call]
    h_color = {1: "#00E68A", -1: "#FF4757", 0: "#FF9F43"}[h_call]
    h_filtered = ""
    if sig.get("filtered"):
        raw_dir = sig.get("signal_raw", "?").upper()
        h_filtered = f"<div style='font-size:10px; color:#FFD93D; margin-top:4px;'>filtered: leaned {raw_dir}, conf below 15%</div>"
    col.markdown(
        f"""
        <div style="background:{h_color}22; border:1px solid {h_color};
                    border-radius:8px; padding:14px; text-align:center;">
            <div style="font-size:11px; color:#888; letter-spacing:1px;">{h}-DAY PREDICTION</div>
            <div style="font-size:24px; font-weight:600; color:{h_color}; margin:6px 0;">{h_label}</div>
            <div style="font-size:12px; color:#aaa;">
                P(up) <b>{sig['proba']:.1%}</b> &nbsp;·&nbsp; conf <b>{sig['confidence']:.0%}</b>
            </div>
            {h_filtered}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption("Signals require ≥75% confidence to act on (P(up) > 0.875 or < 0.125). At this threshold, FLAT is the most common outcome. 30/60-day use the same model architecture but weren't separately tuned/backtested — 10-day is the primary signal.")

# ─── Per-strategy votes ───
st.subheader("Strategy votes")
all_calls = {"xgb (ML)": xgb_call, **current_calls}
strat_cols = st.columns(len(all_calls))
for col, (name, call) in zip(strat_cols, all_calls.items()):
    label = {1: "LONG", -1: "SHORT", 0: "FLAT"}[call]
    color = {1: "#00E68A", -1: "#FF4757", 0: "#FF9F43"}[call]
    is_ml = name.startswith("xgb")
    border_extra = "; box-shadow:0 0 0 1px " + color + "88" if is_ml else ""
    col.markdown(
        f"""
        <div style="background:{color}15; border:1px solid {color}55; border-radius:8px;
                    padding:14px; text-align:center{border_extra}">
            <div style="font-size:11px; color:#888; letter-spacing:0.5px;">{name}</div>
            <div style="font-size:22px; font-weight:600; color:{color}; margin-top:4px;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Context ───
st.subheader("Context")
ctx = st.columns(4)
ctx[0].metric("Regime", current_regime.replace("_", " "),
              help="trending_up/down · ranging · volatile_crash")
rsi_label = "overbought" if current_rsi > 70 else "oversold" if current_rsi < 30 else "neutral"
ctx[1].metric("RSI", f"{current_rsi:.0f}", rsi_label,
              help="<30 oversold · >70 overbought")
ctx[2].metric("vs 20d MA", f"{(last_close/sma20 - 1)*100:+.1f}%",
              help="distance from 20-day moving average")
ctx[3].metric("ADX", f"{current_adx:.0f}", "trending" if current_adx > 25 else "ranging",
              help=">25 = trending · <20 = ranging")

# ─── Options ───
st.subheader("Options play")

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def cached_options_suggest(sym: str, _hash: int, signal: str, conf: float, spot: float, hist_vol: float) -> dict:
    return options.suggest_strategy(
        symbol=sym, spot=spot, signal=signal, confidence=conf,
        horizon_days=10, historical_vol=hist_vol,
    )

historical_vol = float(df["vol_20"].iloc[-1])
if pd.isna(historical_vol) or historical_vol <= 0:
    historical_vol = 0.20

with st.spinner("Fetching options chain & sizing strategy..."):
    try:
        opt_result = cached_options_suggest(
            symbol, len(df), xgb_sig["signal"],
            float(xgb_sig["confidence"]), float(last_close), historical_vol,
        )
    except Exception as e:
        st.warning(f"Options suggestion failed: {e}")
        opt_result = None

if opt_result is not None:
    strat = opt_result["strategy"]
    is_credit = strat.net_debit < 0
    cost_label = (
        f"Collect ${-strat.net_debit:.2f}/share" if is_credit
        else f"Pay ${strat.net_debit:.2f}/share"
    )

    if is_credit:
        strat_color = "#4FACFE"
    elif xgb_call > 0:
        strat_color = "#00E68A"
    elif xgb_call < 0:
        strat_color = "#FF4757"
    else:
        strat_color = "#FF9F43"

    iv_pct = (opt_result["iv_used"] or 0) * 100
    em_pct = opt_result["expected_move_pct"]
    exp_label = opt_result["expiration"] or "no listed expirations"

    st.markdown(
        f"""
        <div style="background:{strat_color}22; border:2px solid {strat_color};
                    border-radius:12px; padding:18px; margin:10px 0;">
            <div style="font-size:24px; font-weight:700; color:{strat_color};">{strat.name}</div>
            <div style="font-size:11px; color:#888; margin-top:4px;">
                exp <b>{exp_label}</b> ({strat.days_to_expiry}d) · IV <b>{iv_pct:.1f}%</b> · expected move <b>{em_pct:+.1f}%</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cost_cols = st.columns(4)
    cost_cols[0].metric("Cost", cost_label, help="× 100 = per contract")
    if strat.max_profit == float("inf"):
        cost_cols[1].metric("Max profit", "unlimited")
    else:
        cost_cols[1].metric("Max profit", f"${strat.max_profit:.2f}/sh",
                            help=f"${strat.max_profit*100:.0f} per contract")
    cost_cols[2].metric("Max loss", f"${strat.max_loss:.2f}/sh",
                        help=f"${strat.max_loss*100:.0f} per contract")
    if len(strat.breakevens) == 1:
        delta_be = (strat.breakevens[0] / last_close - 1) * 100
        cost_cols[3].metric("Breakeven", f"${strat.breakevens[0]:.2f}", f"{delta_be:+.1f}%")
    elif len(strat.breakevens) == 2:
        cost_cols[3].metric("Profit zone", f"${strat.breakevens[0]:.0f}–${strat.breakevens[1]:.0f}")

    legs_str = " · ".join(
        f"**{('BUY' if l.side == 'buy' else 'SELL')}** ${l.strike:.2f} {l.type[0].upper()} @ ${l.premium:.2f}"
        for l in strat.legs
    )
    st.markdown(legs_str)

    payoff_df = options.payoff_diagram(strat, last_close)
    st.line_chart(payoff_df, height=200, use_container_width=True)

    for w in opt_result.get("warnings", []):
        st.warning(w)

    with st.expander("Greeks (primary leg)"):
        primary = strat.legs[0]
        T_years = strat.days_to_expiry / 365
        greeks = options.bs_greeks(
            S=last_close, K=primary.strike, T=T_years,
            r=options.DEFAULT_RISK_FREE_RATE,
            sigma=opt_result["iv_used"] or historical_vol,
            option_type=primary.type,
        )
        gcols = st.columns(5)
        gcols[0].metric("Δ", f"{greeks['delta']:+.3f}", help="per $1 underlying move")
        gcols[1].metric("Γ", f"{greeks['gamma']:.4f}", help="Δ change per $1")
        gcols[2].metric("Θ/d", f"${greeks['theta']:.3f}", help="daily decay")
        gcols[3].metric("V/1%", f"${greeks['vega']:.3f}", help="per 1pp IV")
        gcols[4].metric("ρ/1%", f"${greeks['rho']:.4f}", help="per 1pp rate")
else:
    st.info("No options chain (e.g. crypto). Spot signal still valid.")

# ─── Historical performance ───
st.subheader("Backtest")
if xgb_metrics is not None and xgb_metrics["trades"] > 0:
    bcols = st.columns(4)
    win_pct = xgb_metrics["win_rate"] * 100
    win_color = "normal" if win_pct >= 50 else "inverse"
    bcols[0].metric("Win rate", f"{win_pct:.1f}%", f"{int(xgb_metrics['trades'])} trades",
                    delta_color=win_color)
    bcols[1].metric("Total return", f"{xgb_metrics['total_return']*100:+.1f}%")
    bcols[2].metric("Sharpe", f"{xgb_metrics['sharpe']:.2f}")
    bcols[3].metric("Max DD", f"{xgb_metrics['max_dd']*100:.1f}%")
else:
    st.warning(f"Backtest unavailable — falling back to rules: {int(m['trades'])} trades, {m['win_rate']*100:.0f}% win.")

# ─── Feature importance ───
with st.expander("Feature importance"):
    importance = xgb_sig.get("importance", {})
    if importance:
        imp_df = pd.DataFrame([{"Feature": k, "Importance": v} for k, v in importance.items()])
        st.bar_chart(imp_df.set_index("Feature"), horizontal=True)

st.caption(
    f"daily bars · 10d horizon · {src_name} · auto-refreshes every 2h · live spot every 30s · not financial advice"
)
