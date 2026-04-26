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

from data import sources
import indicators
import regime
import backtest
import strategies
import ml
import class_xgb
import options

st.set_page_config(page_title="Long or Short?", page_icon="$", layout="centered")

st.title("Long or Short?")

col_sym, col_speed = st.columns([3, 2])
with col_sym:
    symbol = st.text_input(
        "Symbol", value="SPY",
        help="e.g. AAPL, SPY, QQQ, BTC-USD",
    ).upper().strip()
with col_speed:
    model_speed = st.radio(
        "Model",
        options=["Fast (XGB)", "Best (5-model ensemble)"],
        index=1,
        help="Fast: 1 model, ~2s. Best: avg of 5 models, ~10s, +0.07 Sharpe.",
    )
use_ensemble = model_speed.startswith("Best")

if not st.button("Get recommendation", type="primary", use_container_width=True):
    st.stop()

if not symbol:
    st.error("Enter a stock symbol.")
    st.stop()


@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
def cached_pooled_today(class_name: str, start_iso: str, ensemble_flag: bool) -> dict:
    if ensemble_flag:
        return class_xgb.predict_today_for_class_ensemble(class_name, start_iso)
    return class_xgb.predict_today_for_class(class_name, start_iso)


@st.cache_data(show_spinner=False)
def cached_pooled_walkforward(class_name: str, start_iso: str, ensemble_flag: bool) -> dict:
    if ensemble_flag:
        return class_xgb.predict_class_ensemble(
            class_name, start_iso, horizon=10, train_size=750, step_size=60
        )
    return class_xgb.predict_class(
        class_name, start_iso, horizon=10, train_size=750, step_size=60
    )


@st.cache_data(show_spinner=False)
def cached_per_symbol_today(sym: str, _df_hash: int, ensemble_flag: bool) -> dict:
    if ensemble_flag:
        predictor = ml.EnsemblePredictor().fit(df)
    else:
        predictor = ml.XGBPredictor().fit(df)
    sig = predictor.predict_now(df)
    sig["importance"] = predictor.feature_importance().head(8).to_dict()
    return sig


@st.cache_data(show_spinner=False)
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

# ─── Big recommendation card (XGBoost is the headline) ───
action_label = {1: "BUY LONG", -1: "SELL SHORT", 0: "STAY FLAT"}[recommendation_call]
action_color = {1: "#00E68A", -1: "#FF4757", 0: "#FF9F43"}[recommendation_call]

st.markdown(
    f"""
    <div style="background:{action_color}22; border:2px solid {action_color};
                border-radius:12px; padding:24px; text-align:center; margin:16px 0;">
        <div style="font-size:11px; color:#888; letter-spacing:1.5px;">10-DAY PREDICTION</div>
        <div style="font-size:42px; font-weight:700; color:{action_color}; margin:8px 0;">{action_label}</div>
        <div style="font-size:13px; color:#aaa;">
            P(up) <b>{xgb_sig['proba']:.1%}</b> · conf <b>{xgb_sig['confidence']:.0%}</b>
        </div>
        <div style="font-size:10px; color:#666; margin-top:6px;">{model_label}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

@st.cache_data(show_spinner=False)
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

st.caption(f"daily bars · 10d horizon · {src_name} · not financial advice")
