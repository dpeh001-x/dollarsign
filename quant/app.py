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
<<<<<<< Updated upstream
import backtest
=======
import class_xgb
import options
import social_scanner
import screener
import trend_momentum
import trending_advisor
import signals as signals_mod
import events as events_mod
import calibration as calibration_mod
import signal_log as signal_log_mod
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
st.caption("US + Hong Kong stocks · ML signal + Wall Street consensus")
=======
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
    def cached_screen(ensemble_flag: bool, bucket: int) -> list[dict]:
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
            # Classify every row into the tier ladder using model probability alone.
            for r in screen_rows:
                idx = signals_mod._proba_to_tier_idx(r["proba"])
                tier_key, tier_label, tier_color, tier_dir = signals_mod.TIERS[idx]
                r["tier_idx"] = idx
                r["tier_key"] = tier_key
                r["tier_label"] = tier_label
                r["tier_color"] = tier_color
                r["tier_dir"] = tier_dir

            longs = sorted(
                [r for r in screen_rows if r["tier_dir"] > 0],
                key=lambda r: r["proba"], reverse=True,
            )
            shorts = sorted(
                [r for r in screen_rows if r["tier_dir"] < 0],
                key=lambda r: r["proba"],
            )
            strong_long = [r for r in longs if r["tier_key"] == "STRONG_LONG"]
            strong_short = [r for r in shorts if r["tier_key"] == "STRONG_SHORT"]

            def _render_screener_table(rows: list[dict]) -> None:
                df_t = pd.DataFrame([{
                    "Ticker": r["ticker"],
                    "Tier": r["tier_label"],
                    "P(up)": r["proba"],
                    "Edge": abs(r["proba"] - 0.5) * 100,
                    "Class": r["class"],
                    "Price": r["last_close"],
                } for r in rows])
                st.dataframe(
                    df_t, use_container_width=True, hide_index=True,
                    column_config={
                        "P(up)": st.column_config.NumberColumn(format="%.1f%%"),
                        "Edge": st.column_config.NumberColumn(format="%.1f pp", help="Percentage points away from 50/50"),
                        "Price": st.column_config.NumberColumn(format="$%.2f"),
                    },
                )

            # ── Headline section: prefer STRONG; if none, fall back to next-best ──
            if strong_long or strong_short:
                if strong_long:
                    st.markdown(f"### STRONG BUY candidates ({len(strong_long)})  —  P(up) ≥ 65%")
                    _render_screener_table(strong_long[:10])
                    top = ", ".join(r["ticker"] for r in strong_long[:5])
                    st.caption(f"Drop one into the Symbol input below for full analysis. Top: **{top}**")
                if strong_short:
                    st.markdown(f"### STRONG SHORT candidates ({len(strong_short)})  —  P(up) < 35%")
                    _render_screener_table(strong_short[:10])
                    top = ", ".join(r["ticker"] for r in strong_short[:5])
                    st.caption(f"Top: **{top}**")
            else:
                # No STRONG — surface the next-best directional leans instead of giving up.
                long_side = [r for r in longs if r["tier_key"] in ("LONG", "LEAN_LONG")][:10]
                short_side = [r for r in shorts if r["tier_key"] in ("SHORT", "LEAN_SHORT")][:10]
                st.info(
                    f"No symbol cleared the STRONG threshold (P(up) ≥ 65% or ≤ 35%) right now — "
                    f"the model isn't seeing a high-confidence setup across the {len(screen_rows)}-symbol universe. "
                    f"That's normal — most days the answer is to wait. "
                    f"The strongest current leans are shown below; cross-reference with the "
                    f"**Trend & momentum tracker** panel to find ones where technicals confirm."
                )
                if long_side:
                    st.markdown(f"##### Strongest LONG leans ({len(long_side)})")
                    _render_screener_table(long_side)
                if short_side:
                    st.markdown(f"##### Strongest SHORT leans ({len(short_side)})")
                    _render_screener_table(short_side)
                if not long_side and not short_side:
                    st.warning(
                        "Every symbol is reading FLAT (P(up) between 48% and 52%). "
                        "The market is genuinely undecided — best move is to sit out today."
                    )

            with st.expander(f"All {len(screen_rows)} results (sorted by edge)"):
                all_sorted = sorted(screen_rows, key=lambda r: abs(r["proba"] - 0.5), reverse=True)
                df_all = pd.DataFrame([{
                    "Ticker": r["ticker"],
                    "Class": r["class"],
                    "Tier": r["tier_label"],
                    "P(up)": r["proba"],
                    "Edge": abs(r["proba"] - 0.5) * 100,
                    "Price": r["last_close"],
                } for r in all_sorted])
                st.dataframe(
                    df_all, use_container_width=True, hide_index=True,
                    column_config={
                        "P(up)": st.column_config.NumberColumn(format="%.1f%%"),
                        "Edge": st.column_config.NumberColumn(format="%.1f pp"),
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
    def cached_trend_momentum(bucket: int) -> list[dict]:
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

# ─── Signal log + live hit rate (the diagnostic loop) ───
with st.expander("Signal log & live hit rate", expanded=False):
    st.caption(
        "Every tier verdict the app produces is appended to a JSONL log "
        "(`quant/signal_log.jsonl`). Once each signal's 10-day horizon elapses, "
        "the actual price action is replayed against the suggested stop/TP to "
        "determine the outcome. Live hit rate = empirical proof that the "
        "model + tier system is working out-of-sample, NOT in-backtest."
    )

    log_df_raw = signal_log_mod.load_log()
    if log_df_raw.empty:
        st.info(
            "No signals logged yet. Every time you click **Get recommendation** "
            "for a symbol, the produced verdict is logged here. Come back after "
            "a few weeks to see the live hit rate."
        )
    else:
        log_col1, log_col2 = st.columns([3, 1])
        log_col1.write(
            f"**{len(log_df_raw)} signals logged.** Click to resolve outcomes "
            f"against actual price action (replays stops/TPs bar-by-bar)."
        )
        do_resolve = log_col2.button("Resolve outcomes", use_container_width=True, key="resolve_log")

        if do_resolve or st.session_state.get("log_resolved_df") is not None:
            if do_resolve:
                with st.spinner("Resolving outcomes against historical bars..."):
                    try:
                        def _get_bars(sym: str, start: str) -> pd.DataFrame:
                            return sources.get_bars(sym, start, use_cache=True)
                        resolved = signal_log_mod.resolve_outcomes(log_df_raw, _get_bars)
                        st.session_state["log_resolved_df"] = resolved
                    except Exception as e:
                        st.error(f"Resolution failed: {e}")
                        st.session_state["log_resolved_df"] = log_df_raw.assign(
                            resolved=False, outcome="error", pnl_pct=None,
                        )
            resolved = st.session_state.get("log_resolved_df", log_df_raw)
            stats = signal_log_mod.live_stats(resolved)

            # ── Headline metrics ──
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total signals", stats.get("total", 0))
            m2.metric("Resolved", stats.get("resolved", 0))
            m3.metric("Pending", stats.get("pending", 0))
            wr = stats.get("win_rate_overall")
            if wr is not None:
                m4.metric("Live win rate", f"{wr*100:.1f}%",
                          help=f"Across {stats.get('non_flat_resolved', 0)} non-flat resolved signals")
            else:
                m4.metric("Live win rate", "—", help="No resolved non-flat signals yet")

            # ── Hit rate by tier (compare to calibrator's promised band) ──
            by_tier = stats.get("by_tier")
            if by_tier is not None and not by_tier.empty:
                st.markdown("##### Hit rate by tier")
                st.dataframe(
                    by_tier.rename(columns={"n": "N", "wins": "Wins", "win_rate": "Win rate", "avg_pnl": "Avg PnL"}),
                    use_container_width=True, hide_index=True,
                    column_config={
                        "Win rate": st.column_config.NumberColumn(format="%.1f%%"),
                        "Avg PnL": st.column_config.NumberColumn(format="%+.2f%%"),
                    },
                )

            by_conv = stats.get("by_conviction")
            if by_conv is not None and not by_conv.empty:
                st.markdown("##### Hit rate by conviction (★)")
                st.dataframe(
                    by_conv.rename(columns={"n": "N", "wins": "Wins", "win_rate": "Win rate", "avg_pnl": "Avg PnL"}),
                    use_container_width=True, hide_index=True,
                    column_config={
                        "Win rate": st.column_config.NumberColumn(format="%.1f%%"),
                        "Avg PnL": st.column_config.NumberColumn(format="%+.2f%%"),
                    },
                )

            st.markdown("##### Last 25 signals")
            recent_cols = ["ts", "symbol", "entry_date", "entry_price", "label",
                           "conviction", "proba_raw", "outcome", "pnl_pct"]
            available = [c for c in recent_cols if c in resolved.columns]
            recent = resolved.sort_values("ts", ascending=False).head(25)[available]
            st.dataframe(
                recent, use_container_width=True, hide_index=True,
                column_config={
                    "entry_price": st.column_config.NumberColumn(format="$%.2f"),
                    "proba_raw": st.column_config.NumberColumn(format="%.1f%%"),
                    "pnl_pct": st.column_config.NumberColumn(format="%+.2f%%"),
                },
            )
        else:
            st.caption("Click **Resolve outcomes** to score the log against actual price action.")
            recent_cols = ["ts", "symbol", "entry_date", "entry_price", "label",
                           "conviction", "proba_raw"]
            available = [c for c in recent_cols if c in log_df_raw.columns]
            recent = log_df_raw.sort_values("ts", ascending=False).head(25)[available]
            st.dataframe(
                recent, use_container_width=True, hide_index=True,
                column_config={
                    "entry_price": st.column_config.NumberColumn(format="$%.2f"),
                    "proba_raw": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )

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
>>>>>>> Stashed changes


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

<<<<<<< Updated upstream
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
=======
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
def cached_per_symbol_today(sym: str, df_hash: int, ensemble_flag: bool) -> dict:
    if ensemble_flag:
        predictor = ml.EnsemblePredictor().fit(df)
    else:
        predictor = ml.XGBPredictor().fit(df)
    sig = predictor.predict_now(df)
    sig["importance"] = predictor.feature_importance().head(8).to_dict()
    return sig


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def cached_per_symbol_walkforward(sym: str, df_hash: int, ensemble_flag: bool) -> dict | None:
    """Returns dict with keys 'metrics' (for backtest card) and 'proba' (pd.Series
    of out-of-sample P(up), for calibration). None on failure."""
    try:
        if ensemble_flag:
            wf = ml.EnsembleWalkForward()
            proba = wf.fit_predict(df)
            positions = ml.proba_to_positions(proba, wf.long_thresh, wf.short_thresh)
            positions = positions.reindex(df.index, fill_value=0.0).fillna(0.0)
        else:
            wf = ml.WalkForwardModel()
            proba = wf.fit_predict(df)
            positions = ml.proba_to_positions(proba, wf.long_thresh, wf.short_thresh)
            positions = positions.reindex(df.index, fill_value=0.0).fillna(0.0)
        metrics = backtest.run(df, positions, fee_bps=1.0, slippage_bps=1.0).metrics
        return {"metrics": metrics, "proba": proba}
    except Exception:
        return None


# ─── Multi-horizon prediction helpers ───
@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def cached_pooled_today_horizon(class_name: str, start_iso: str, ensemble_flag: bool, horizon: int) -> dict:
    if ensemble_flag:
        return class_xgb.predict_today_for_class_ensemble(class_name, start_iso, horizon=horizon)
    return class_xgb.predict_today_for_class(class_name, start_iso, horizon=horizon)


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def cached_per_symbol_today_horizon(sym: str, df_hash: int, ensemble_flag: bool, horizon: int) -> dict:
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

# ─── Compute trend + momentum scores from the already-loaded df ───
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

# ─── Rule-based strategies (secondary context) ───
positions_by_strategy = {
    "regime_breakout": strategies.regime_aware_breakout(df),
    "rsi_meanrev": strategies.rsi_mean_reversion(df),
    "ma_cross": strategies.ma_crossover(df),
}
current_calls = {name: int(p.iloc[-1]) for name, p in positions_by_strategy.items()}

# Rule-based ensemble metrics (kept for the fallback branch only)
ensemble_series = (
    sum(positions_by_strategy.values())
    .apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
)
m = backtest.run(df, ensemble_series, fee_bps=1.0, slippage_bps=1.0).metrics

# ─── XGB walk-forward → backtest metrics + calibration-fit data ───
xgb_metrics = None
wf_proba: pd.Series | None = None
if use_pooled:
    kind = "5-model ensemble" if use_ensemble else "XGBoost"
    with st.spinner(f"Walk-forward backtesting pooled {kind} on {sym_class}..."):
        try:
            probas_by_sym = cached_pooled_walkforward(sym_class, start_date_iso, use_ensemble)
            wf_proba = probas_by_sym.get(symbol)
            if wf_proba is not None and not wf_proba.dropna().empty:
                proba_aligned = wf_proba.reindex(df.index)
                positions = pd.Series(0.0, index=df.index)
                positions[proba_aligned > 0.55] = 1.0
                positions[proba_aligned < 0.45] = -1.0
                xgb_metrics = backtest.run(df, positions, fee_bps=1.0, slippage_bps=1.0).metrics
        except Exception:
            xgb_metrics = None
else:
    kind = "5-model ensemble" if use_ensemble else "XGBoost"
    with st.spinner(f"Walk-forward backtesting per-symbol {kind}..."):
        wf_result = cached_per_symbol_walkforward(symbol, len(df), use_ensemble)
        if wf_result:
            xgb_metrics = wf_result.get("metrics")
            wf_proba = wf_result.get("proba")

# ─── Fit isotonic calibration on the walk-forward predictions ───
calibrator = None
if wf_proba is not None and not wf_proba.dropna().empty:
    try:
        outcomes = calibration_mod.compute_outcomes(df["close"], horizon=10)
        # Align indices
        df_cal = pd.concat([wf_proba.rename("p"), outcomes.rename("y")], axis=1).dropna()
        if len(df_cal) >= 30:
            calibrator = calibration_mod.fit_calibration(df_cal["p"], df_cal["y"], n_bands=5)
    except Exception:
        calibrator = None

raw_proba = float(xgb_sig["proba"])
if calibrator is not None:
    cal_proba = calibrator.calibrate(raw_proba)
    hit_rate, hit_rate_n = calibrator.hit_rate_at(raw_proba)
else:
    cal_proba = None
    hit_rate = None
    hit_rate_n = None

# ─── Earnings blackout check ───
try:
    earnings = events_mod.check_earnings_smart(symbol, blackout_window=5)
except Exception:
    earnings = events_mod.EarningsCheck(
        has_date=False, earnings_date=None, trading_days_until=None,
        in_blackout=False, blackout_window=5,
    )

# ─── Tier classification with all the upgrades ───
trade_signal = signals_mod.classify_signal(
    proba=raw_proba,
    trend_score=_trend_score, momentum_score=_momentum_score,
    adx=float(df["adx"].iloc[-1]) if "adx" in df.columns else float("nan"),
    rsi=float(df["rsi_14"].iloc[-1]),
    calibrated_proba=cal_proba,
    empirical_hit_rate=hit_rate,
    hit_rate_n_samples=hit_rate_n,
    earnings_in_days=earnings.trading_days_until if earnings.has_date else None,
    blackout_window=5,
)
xgb_call = trade_signal.direction  # -1, 0, +1
# Keep legacy fields populated for the strategy-votes display
xgb_sig["signal"] = {1: "long", -1: "short", 0: "flat"}[xgb_call]

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
def cached_live_quote(sym: str, bucket: int) -> dict | None:
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

# ─── Earnings warning banner (above the verdict card if blackout active) ───
if ts.in_earnings_blackout and earnings.has_date:
    st.markdown(
        f"""
        <div style="background:#FF475722; border:2px solid #FF4757;
                    border-radius:8px; padding:12px 16px; margin:12px 0 -4px 0;">
            <div style="font-size:11px; color:#FF4757; letter-spacing:1.5px; font-weight:700;">⚠ EARNINGS BLACKOUT</div>
            <div style="font-size:13px; color:#fff; margin-top:4px;">
                {symbol} reports earnings in <b>{earnings.trading_days_until} trading day{'s' if earnings.trading_days_until != 1 else ''}</b>
                ({earnings.earnings_date.strftime('%b %d, %Y')}). The directional signal has been
                <b>vetoed to FLAT</b> — technical models don't survive EPS surprises and gap risk
                makes stops unreliable. Consider waiting until after the report.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Compose the probability display (raw + calibrated when available)
if ts.calibrated_proba is not None and abs(ts.calibrated_proba - ts.proba) > 0.02:
    proba_html = (
        f"P(up) raw <b>{ts.proba:.1%}</b> "
        f"<span style='color:#FFD93D'>→ calibrated <b>{ts.calibrated_proba:.1%}</b></span>"
    )
else:
    proba_html = f"P(up) <b>{ts.proba:.1%}</b>"

# Hit-rate annotation
hit_rate_html = ""
if ts.empirical_hit_rate is not None and ts.hit_rate_n_samples and ts.hit_rate_n_samples >= 20:
    hit_color = "#00E68A" if ts.empirical_hit_rate >= 0.55 else "#FF9F43" if ts.empirical_hit_rate >= 0.45 else "#FF4757"
    hit_rate_html = (
        f"<div style='text-align:center; font-size:11px; color:#888; margin-top:4px;'>"
        f"historical hit rate at this confidence: "
        f"<b style='color:{hit_color}'>{ts.empirical_hit_rate:.0%}</b> "
        f"<span style='color:#666'>(n={ts.hit_rate_n_samples})</span>"
        f"</div>"
    )

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
            {proba_html}
            &nbsp;·&nbsp; trend <b>{ts.trend_score:+.2f}</b>
            &nbsp;·&nbsp; momentum <b>{ts.momentum_score:+.2f}</b>
            &nbsp;·&nbsp; confluence <b style="color:{conf_color};">{conf_str}</b>
        </div>
        {hit_rate_html}
        <ul style="margin:14px 0 4px 0; padding-left:22px;">
            {reasoning_html}
        </ul>
        <div style="font-size:10px; color:#666; margin-top:8px; text-align:right;">{model_label}</div>
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
          </div>
        """
=======
            """,
            unsafe_allow_html=True,
        )

# ─── Persist this signal to the JSONL log (deduped per symbol/date/tier/conviction) ───
try:
    sizing_obj = sizing if (xgb_call != 0 and 'sizing' in locals() and sizing is not None) else None
    _written = signal_log_mod.log_signal(
        symbol=symbol,
        entry_date=last_date,
        entry_price=float(last_close),
        trade_signal=ts,
        sizing=sizing_obj,
        horizon_days=10,
        model_kind=model_label,
    )
except Exception as _e:
    # Logging must never break the app
    pass

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
def cached_options_suggest(sym: str, data_hash: int, signal: str, conf: float, spot: float, hist_vol: float) -> dict:
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
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
=======
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

# ─── Probability cone (Monte Carlo forecast) ───
st.subheader("Probability cone — 30 days ahead")

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def cached_cone(sym: str, signal: str, conf_rounded: float, df_hash: int) -> tuple:
    # Raise on failure (exceptions are NOT cached — a transient error won't
    # pin a None for the whole TTL). Return PNG bytes, not a path, so a
    # later overwrite of the file can't corrupt what this cache serves.
    import forecast
    cone, summary, png = forecast.forecast_ticker(sym, signal, conf_rounded)
    return summary, Path(png).read_bytes()

try:
    cone_result = cached_cone(symbol, xgb_sig["signal"], round(float(xgb_sig["confidence"]), 2), len(df))
except Exception:
    cone_result = None
if cone_result:
    cone_summary_d, cone_png = cone_result
    st.image(cone_png, use_container_width=True)
    st.caption(
        f"10,000 simulated paths · drift tilted by the live-calibrated edge · "
        f"median at day 10: {cone_summary_d['median_pct']:+.1f}% · "
        f"90% of simulations land between {cone_summary_d['p5_pct']:+.1f}% and {cone_summary_d['p95_pct']:+.1f}%. "
        f"This is a probability distribution, not a prediction."
    )

    # Live calibration state (the self-improvement loop)
    try:
        import signal_log
        jstats = signal_log.journal_stats()
        if jstats["total_signals"] > 0:
            live_wr = (
                f"{jstats['live_win_rate']*100:.1f}% on {jstats['scored']} scored"
                if jstats["live_win_rate"] is not None else f"accumulating ({jstats['open']} open)"
            )
            st.caption(
                f"Self-calibration journal: {jstats['total_signals']} signals logged · live win rate: {live_wr} · "
                f"win-rate estimates auto-update as signals mature."
            )
    except Exception:
        pass

# ─── Feature importance ───
with st.expander("Feature importance"):
    importance = xgb_sig.get("importance", {})
    if importance:
        imp_df = pd.DataFrame([{"Feature": k, "Importance": v} for k, v in importance.items()])
        st.bar_chart(imp_df.set_index("Feature"), horizontal=True)

>>>>>>> Stashed changes
st.caption(
    "BUY/SELL only fires when ALL gates pass: model confidence ≥25%, verified historical "
    "edge on this exact symbol (≥55% win rate, positive Sharpe), trend alignment, and "
    "analyst-consensus agreement. Risk max 1–2% of your account per trade and always place "
    "the stop-loss order. Not financial advice. Past performance does not guarantee future results."
)
