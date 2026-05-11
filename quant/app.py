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
import trend_momentum
import trending_advisor
import signals as signals_mod

st.set_page_config(page_title="Long or Short?", page_icon="$", layout="centered")

st.title("Long or Short?")
st.caption("XGBoost predicts next-10-day direction · 7-tier signal (STRONG_SHORT → STRONG_LONG) with technical confluence · Structural stops + conviction-scaled sizing")

# ─── Risk / sizing settings (sidebar) ───
with st.sidebar:
    st.header("Trade settings")
    st.caption("Used to size the suggested position. Stays in this browser session only.")
    account_size = st.number_input(
        "Account size ($)", min_value=100.0, max_value=10_000_000.0,
        value=float(st.session_state.get("account_size", 10000.0)),
        step=500.0, format="%.0f",
        help="Total trading capital. Position values + risk are computed against this.",
    )
    base_risk_pct = st.slider(
        "Base risk per trade (%)",
        min_value=0.25, max_value=3.0,
        value=float(st.session_state.get("base_risk_pct", 1.0)),
        step=0.25,
        help="Risked on a 5★ conviction setup. Lower-conviction setups risk proportionally less.",
    ) / 100.0
    st.session_state["account_size"] = account_size
    st.session_state["base_risk_pct"] = base_risk_pct * 100

    st.caption(
        f"5★ trade risks **${account_size * base_risk_pct:,.0f}** ({base_risk_pct*100:.2f}%). "
        f"1★ risks **${account_size * base_risk_pct * 0.2:,.0f}** ({base_risk_pct*100*0.2:.2f}%)."
    )
    st.divider()
    st.caption("Not financial advice. Past backtest performance does not guarantee future results.")

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
    st.caption("Live cross-platform scan via apewisdom.io (aggregates Reddit + StockTwits + Twitter). Each ticker is then scored on trend & momentum and given an objective BUY / ACCUMULATE / WATCH / AVOID / SHORT verdict with a thesis.")

    sc1, sc2 = st.columns([2, 2])
    filt = sc1.selectbox(
        "Source",
        options=list(social_scanner.VALID_FILTERS),
        index=0,
        help="all-stocks: combined feed. wallstreetbets/options/etc: source-specific. cryptos: BTC/ETH/etc.",
    )
    @st.cache_data(show_spinner=False, ttl=300)
    def cached_trending_advice(filter_name: str, _ts: int, max_advice: int = 15) -> list[dict]:
        sigs = social_scanner.scan(limit=max(25, max_advice), filter_name=filter_name)
        advised = trending_advisor.advise(sigs, max_advice=max_advice)
        return [{
            "ticker": a.ticker, "company": a.company_name,
            "mentions": a.mentions, "yesterday": a.mentions_24h_ago,
            "social_momentum": a.social_momentum, "social_score": a.social_score,
            "last_close": a.last_close,
            "trend_score": a.trend_score, "trend_label": a.trend_label,
            "momentum_score": a.momentum_score, "momentum_label": a.momentum_label,
            "rsi_14": a.rsi_14, "adx": a.adx,
            "ret_5d": a.ret_5d, "ret_20d": a.ret_20d,
            "verdict": a.verdict, "verdict_color": a.verdict_color,
            "conviction": a.conviction, "thesis": a.thesis,
        } for a in advised]

    if sc2.button("Scan now", use_container_width=True):
        with st.spinner(f"Scanning apewisdom + scoring top tickers ({filt})..."):
            try:
                import time as _t
                rows = cached_trending_advice(filt, int(_t.time() // 300))
            except Exception as e:
                st.error(f"Scan failed: {e}")
                rows = []

        if rows:
            # ─── Summary table with verdict column ───
            df_st = pd.DataFrame([{
                "Ticker": r["ticker"],
                "Verdict": r["verdict"],
                "Conv": "★" * r["conviction"],
                "Trend": (r["trend_label"] or "—").replace("_", " "),
                "Momentum": (r["momentum_label"] or "—").replace("_", " "),
                "T score": r["trend_score"],
                "M score": r["momentum_score"],
                "RSI": r["rsi_14"],
                "20d %": (r["ret_20d"] * 100) if r["ret_20d"] is not None else None,
                "Social Mom×": r["social_momentum"],
                "Mentions": r["mentions"],
            } for r in rows])
            st.dataframe(
                df_st, use_container_width=True, hide_index=True,
                column_config={
                    "Verdict": st.column_config.TextColumn("Verdict", width="small"),
                    "Conv": st.column_config.TextColumn("Conv", width="small", help="Conviction 1–5"),
                    "T score": st.column_config.NumberColumn("Trend", format="%+.2f", help="−1..+1"),
                    "M score": st.column_config.NumberColumn("Mom", format="%+.2f", help="−1..+1"),
                    "RSI": st.column_config.NumberColumn(format="%.0f"),
                    "20d %": st.column_config.NumberColumn(format="%+.1f%%"),
                    "Social Mom×": st.column_config.NumberColumn(format="%.1fx", help="mentions today / yesterday"),
                },
            )

            # ─── Verdict-coloured cards with thesis ───
            st.markdown("##### Per-ticker thesis")
            for r in rows:
                price_str = f"${r['last_close']:.2f}" if r["last_close"] else "—"
                co = r["company"] or ""
                co_str = f" · <span style='color:#888'>{co}</span>" if co else ""
                st.markdown(
                    f"""
                    <div style="background:{r['verdict_color']}15;
                                border-left:4px solid {r['verdict_color']};
                                border-radius:6px; padding:10px 14px; margin:6px 0;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <b style="font-size:15px; color:#fff;">{r['ticker']}</b>
                                <span style="color:#aaa; margin-left:6px;">{price_str}</span>
                                {co_str}
                            </div>
                            <div>
                                <span style="background:{r['verdict_color']}; color:#0E1117;
                                             padding:3px 10px; border-radius:4px;
                                             font-weight:700; font-size:12px;">{r['verdict']}</span>
                                <span style="color:{r['verdict_color']}; margin-left:8px; font-size:13px;">
                                    {'★' * r['conviction']}<span style="color:#333">{'★' * (5 - r['conviction'])}</span>
                                </span>
                            </div>
                        </div>
                        <div style="margin-top:6px; font-size:12px; color:#ccc; line-height:1.5;">
                            {r['thesis']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            top_buys = [r["ticker"] for r in rows if r["verdict"] == "BUY"][:5]
            if top_buys:
                st.caption(f"Highest-conviction BUYs from trending: **{', '.join(top_buys)}** — drop one into the Symbol box below for the full ML analysis.")
            else:
                st.caption("No BUY verdicts in the current trending set. Drop any ticker into the Symbol box below for the full ML analysis anyway.")
            st.caption("⚠ Verdicts are objective rule-based scoring, not financial advice. The crowd is often wrong; size positions accordingly.")
        elif rows == []:
            st.info("No tickers returned.")

# ─── Stock screener (find actionable signals across the universe) ───
with st.expander("Stock screener — find high-conviction signals", expanded=False):
    st.caption(f"Scans {screener.SCREEN_UNIVERSE_SIZE} symbols across mega-caps, broad ETFs, sector ETFs, industry ETFs, macro/commodity ETFs, international ETFs, and crypto. Surfaces only those meeting the confidence threshold (currently {0.75:.0%}).")

    @st.cache_data(show_spinner=False, ttl=15 * 60)
    def cached_screen(ensemble_flag: bool, _bucket: int) -> list[dict]:
        rows = screener.screen(use_ensemble=ensemble_flag)
        return [{
            "ticker": r.ticker, "class": r.asset_class,
            "signal": r.signal, "proba": r.proba,
            "confidence": r.confidence, "last_close": r.last_close,
        } for r in rows]

    sc_col1, sc_col2 = st.columns([3, 1])
    sc_col1.write(f"**Click to scan** {screener.SCREEN_UNIVERSE_SIZE} symbols — first run takes ~5-10 min (ensemble) or ~2 min (Fast mode); cached for 15 min afterward.")
    if sc_col2.button("Run screen", use_container_width=True, key="run_screen"):
        st.session_state["screen_run"] = True

    if st.session_state.get("screen_run"):
        with st.spinner("Scanning 25 symbols..."):
            try:
                import time as _t
                screen_rows = cached_screen(use_ensemble, int(_t.time() // (15 * 60)))
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

# ─── Trend & momentum tracker (pure indicators, no ML) ───
with st.expander("Trend & momentum tracker", expanded=False):
    st.caption(
        f"Indicator-based scan of {screener.SCREEN_UNIVERSE_SIZE} symbols. "
        "Trend = SMA50/SMA200 alignment + price vs SMA200 + ADX×DI direction. "
        "Momentum = 20d/5d returns + RSI distance from 50 + MACD histogram. "
        "**Aligned** rows have trend & momentum pointing the same way — strongest directional setups."
    )

    @st.cache_data(show_spinner=False, ttl=15 * 60)
    def cached_trend_momentum(_bucket: int) -> list[dict]:
        rows = trend_momentum.track()
        return [{
            "ticker": r.ticker, "class": r.asset_class,
            "trend_score": r.trend_score, "trend_label": r.trend_label,
            "momentum_score": r.momentum_score, "momentum_label": r.momentum_label,
            "combined_score": r.combined_score, "aligned": r.aligned,
            "adx": r.adx, "rsi_14": r.rsi_14,
            "ret_5d": r.ret_5d, "ret_20d": r.ret_20d,
            "last_close": r.last_close,
        } for r in rows]

    tm_col1, tm_col2 = st.columns([3, 1])
    tm_col1.write(f"**Click to scan.** Uses cached OHLC data — typically <30s. Re-cached for 15 min.")
    if tm_col2.button("Run tracker", use_container_width=True, key="run_tm"):
        st.session_state["tm_run"] = True

    if st.session_state.get("tm_run"):
        with st.spinner(f"Scanning {screener.SCREEN_UNIVERSE_SIZE} symbols for trend + momentum..."):
            try:
                import time as _t
                tm_rows = cached_trend_momentum(int(_t.time() // (15 * 60)))
            except Exception as e:
                st.error(f"Tracker failed: {e}")
                tm_rows = []

        if tm_rows:
            aligned_rows = [r for r in tm_rows if r["aligned"]]

            tab_aligned, tab_trend, tab_mom, tab_all = st.tabs([
                f"Aligned setups ({len(aligned_rows)})",
                "Strongest trends",
                "Strongest momentum",
                f"All {len(tm_rows)}",
            ])

            def _fmt_rows(rows: list[dict]) -> pd.DataFrame:
                return pd.DataFrame([{
                    "Ticker": r["ticker"],
                    "Class": r["class"],
                    "Trend": r["trend_label"].replace("_", " "),
                    "Momentum": r["momentum_label"].replace("_", " "),
                    "Trend score": r["trend_score"],
                    "Mom score": r["momentum_score"],
                    "Combined": r["combined_score"],
                    "RSI": r["rsi_14"],
                    "ADX": r["adx"],
                    "5d %": r["ret_5d"] * 100,
                    "20d %": r["ret_20d"] * 100,
                    "Price": r["last_close"],
                    "Aligned": "yes" if r["aligned"] else "—",
                } for r in rows])

            col_cfg = {
                "Trend score": st.column_config.NumberColumn(format="%+.2f", help="-1..+1"),
                "Mom score": st.column_config.NumberColumn(format="%+.2f", help="-1..+1"),
                "Combined": st.column_config.NumberColumn(format="%+.2f", help="trend + momentum, -2..+2"),
                "RSI": st.column_config.NumberColumn(format="%.0f"),
                "ADX": st.column_config.NumberColumn(format="%.0f", help=">25 = trending"),
                "5d %": st.column_config.NumberColumn(format="%+.1f%%"),
                "20d %": st.column_config.NumberColumn(format="%+.1f%%"),
                "Price": st.column_config.NumberColumn(format="$%.2f"),
            }

            with tab_aligned:
                if aligned_rows:
                    aligned_sorted = sorted(aligned_rows, key=lambda r: abs(r["combined_score"]), reverse=True)
                    st.dataframe(_fmt_rows(aligned_sorted), use_container_width=True,
                                 hide_index=True, column_config=col_cfg)
                    top_up = [r["ticker"] for r in aligned_sorted if r["combined_score"] > 0][:5]
                    top_dn = [r["ticker"] for r in aligned_sorted if r["combined_score"] < 0][:5]
                    if top_up:
                        st.caption(f"Strongest bull setups: **{', '.join(top_up)}**")
                    if top_dn:
                        st.caption(f"Strongest bear setups: **{', '.join(top_dn)}**")
                else:
                    st.info("No symbols with aligned trend + momentum. Market is mixed.")

            with tab_trend:
                trend_sorted = sorted(tm_rows, key=lambda r: abs(r["trend_score"]), reverse=True)[:25]
                st.dataframe(_fmt_rows(trend_sorted), use_container_width=True,
                             hide_index=True, column_config=col_cfg)

            with tab_mom:
                mom_sorted = sorted(tm_rows, key=lambda r: abs(r["momentum_score"]), reverse=True)[:25]
                st.dataframe(_fmt_rows(mom_sorted), use_container_width=True,
                             hide_index=True, column_config=col_cfg)

            with tab_all:
                st.dataframe(_fmt_rows(tm_rows), use_container_width=True,
                             hide_index=True, column_config=col_cfg)

symbol = st.text_input(
    "Symbol", value="SPY",
    help="e.g. AAPL, SPY, QQQ, BTC-USD",
).upper().strip()

# Cache lifetime for data + model predictions. After this, fresh OHLC fetched,
# models retrained. Live spot quote refreshes separately at 30s.
CACHE_TTL_SECONDS = 15 * 60  # 15 minutes
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

# Persist the user's request across auto-refresh re-runs. Every 15 min, page rerun
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
st_autorefresh(interval=AUTO_REFRESH_MS, key="auto_refresh_15m")


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

# ─── New tier classifier (replaces binary 75% confidence filter for 10-day signal) ───
# Compute trend + momentum scores from the already-loaded df.
tm_result = trend_momentum.analyze_from_frame(df, symbol=symbol)
if tm_result is not None:
    _trend_score = tm_result.trend_score
    _momentum_score = tm_result.momentum_score
    _swing_low_20d = float(df["low"].rolling(20).min().iloc[-1])
    _swing_high_20d = float(df["high"].rolling(20).max().iloc[-1])
else:
    _trend_score = 0.0
    _momentum_score = 0.0
    _swing_low_20d = float("nan")
    _swing_high_20d = float("nan")

trade_signal = signals_mod.classify_signal(
    proba=float(xgb_sig["proba"]),
    trend_score=_trend_score, momentum_score=_momentum_score,
    adx=float(df["adx"].iloc[-1]) if "adx" in df.columns else float("nan"),
    rsi=float(df["rsi_14"].iloc[-1]),
)
xgb_call = trade_signal.direction  # -1, 0, +1
# Keep legacy fields populated for the strategy-votes display
xgb_sig["signal"] = {1: "long", -1: "short", 0: "flat"}[xgb_call]

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

# ─── Big recommendation card (tiered classifier output) ───
ts = trade_signal
stars_filled = "★" * ts.conviction
stars_empty = "★" * (5 - ts.conviction)
reasoning_html = "".join(
    f"<li style='margin:3px 0; color:#bbb; font-size:12px;'>{r}</li>"
    for r in ts.reasoning
)
conf_str = f"{ts.confluence}"
conf_color = {
    "aligned": "#00E68A", "opposed": "#FF4757",
    "mixed": "#FF9F43", "neutral": "#888",
}.get(ts.confluence, "#888")

st.markdown(
    f"""
    <div style="background:{ts.color}22; border:2px solid {ts.color};
                border-radius:12px; padding:22px; margin:16px 0;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:11px; color:#888; letter-spacing:1.5px;">10-DAY PREDICTION · TIER</div>
            <div style="font-size:14px; color:{ts.color}; letter-spacing:1px;">
                {stars_filled}<span style="color:#333">{stars_empty}</span>
                <span style="color:#aaa; font-size:11px; margin-left:6px;">conviction {ts.conviction}/5</span>
            </div>
        </div>
        <div style="font-size:38px; font-weight:700; color:{ts.color}; margin:6px 0; text-align:center;">{ts.label}</div>
        <div style="text-align:center; font-size:13px; color:#aaa;">
            P(up) <b>{ts.proba:.1%}</b>
            &nbsp;·&nbsp; trend <b>{ts.trend_score:+.2f}</b>
            &nbsp;·&nbsp; momentum <b>{ts.momentum_score:+.2f}</b>
            &nbsp;·&nbsp; confluence <b style="color:{conf_color};">{conf_str}</b>
        </div>
        <ul style="margin:14px 0 4px 0; padding-left:22px;">
            {reasoning_html}
        </ul>
        <div style="font-size:10px; color:#666; margin-top:8px; text-align:right;">{model_label}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Structural sizing card (replaces fixed 2.5×ATR / 1:1 R:R block) ───
if xgb_call != 0:
    atr_value = float(df["atr_14"].iloc[-1])
    adx_value = float(df["adx"].iloc[-1]) if "adx" in df.columns else float("nan")
    sizing = signals_mod.size_position(
        direction=xgb_call,
        spot=float(last_close),
        atr=atr_value,
        conviction=ts.conviction,
        account_size=float(account_size),
        base_risk_pct=float(base_risk_pct),
        adx=adx_value,
        swing_low_20d=_swing_low_20d,
        swing_high_20d=_swing_high_20d,
        horizon_days=10,
        last_date=last_date,
    )
    if sizing is not None:
        notes_html = "".join(
            f"<div style='font-size:11px; color:#FFD93D; margin-top:4px;'>⚠ {n}</div>"
            for n in sizing.notes
        )
        rp_color = "#FF4757"; tp_color = "#00E68A"
        st.markdown(
            f"""
            <div style="background:#1A1B24; border:1px solid #2A2B38; border-left:4px solid {ts.color};
                        border-radius:8px; padding:14px 16px; margin:-8px 0 16px 0;">
                <div style="font-size:11px; color:#888; letter-spacing:1.5px; margin-bottom:8px;">
                    POSITION SIZING · {sizing.rr_ratio:.0f}:1 R:R
                </div>
                <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:10px; font-size:12px;">
                    <div>
                        <div style="color:#888; font-size:10px;">ENTRY</div>
                        <div style="color:#fff; font-size:16px; font-weight:600;">${sizing.entry:.2f}</div>
                    </div>
                    <div>
                        <div style="color:#888; font-size:10px;">STOP</div>
                        <div style="color:{rp_color}; font-size:16px; font-weight:600;">${sizing.stop:.2f}</div>
                        <div style="color:#666; font-size:10px;">{sizing.pct_to_stop:+.1f}% · risk ${sizing.risk_per_share:.2f}/sh</div>
                    </div>
                    <div>
                        <div style="color:#888; font-size:10px;">TAKE PROFIT</div>
                        <div style="color:{tp_color}; font-size:16px; font-weight:600;">${sizing.take_profit:.2f}</div>
                        <div style="color:#666; font-size:10px;">{sizing.pct_to_tp:+.1f}% · reward ${sizing.reward_per_share:.2f}/sh</div>
                    </div>
                    <div>
                        <div style="color:#888; font-size:10px;">TIME STOP</div>
                        <div style="color:#FFD93D; font-size:16px; font-weight:600;">{sizing.time_stop_date}</div>
                        <div style="color:#666; font-size:10px;">{sizing.horizon_days*3} trading days</div>
                    </div>
                </div>
                <hr style="border:none; border-top:1px solid #2A2B38; margin:12px 0 8px 0;">
                <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:10px; font-size:12px;">
                    <div>
                        <div style="color:#888; font-size:10px;">POSITION SIZE</div>
                        <div style="color:#fff; font-size:16px; font-weight:600;">{sizing.n_shares} sh</div>
                        <div style="color:#666; font-size:10px;">${sizing.position_value:,.0f} ({sizing.position_value/account_size*100:.1f}% of acct)</div>
                    </div>
                    <div>
                        <div style="color:#888; font-size:10px;">RISK</div>
                        <div style="color:{rp_color}; font-size:16px; font-weight:600;">${sizing.risk_dollar:,.0f}</div>
                        <div style="color:#666; font-size:10px;">{sizing.risk_pct_of_account*100:.2f}% of account</div>
                    </div>
                    <div>
                        <div style="color:#888; font-size:10px;">POTENTIAL REWARD</div>
                        <div style="color:{tp_color}; font-size:16px; font-weight:600;">${sizing.n_shares * sizing.reward_per_share:,.0f}</div>
                        <div style="color:#666; font-size:10px;">at TP</div>
                    </div>
                    <div>
                        <div style="color:#888; font-size:10px;">EFFECTIVE RISK %</div>
                        <div style="color:#fff; font-size:16px; font-weight:600;">{sizing.effective_risk_pct*100:.2f}%</div>
                        <div style="color:#666; font-size:10px;">{base_risk_pct*100:.2f}% × {ts.conviction}/5 conv</div>
                    </div>
                </div>
                <div style="font-size:11px; color:#888; margin-top:10px;">
                    <span style="color:#aaa;">Stop basis:</span> {sizing.stop_basis}
                    &nbsp;·&nbsp; <span style="color:#aaa;">TP basis:</span> {sizing.tp_basis}
                </div>
                {notes_html}
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
    f"daily bars · 10d horizon · {src_name} · auto-refreshes every 15 min · live spot every 30s · not financial advice"
)
