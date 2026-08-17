"""Daily automated run — the ongoing loop. Designed for Task Scheduler.

Each run (intended: daily after US market close):
  1. Screen the full universe (fast XGB mode for speed)
  2. Append non-flat signals to the journal
  3. Score any matured signals against what actually happened
  4. Recalibrate the confidence→win-rate map from the live track record
  5. Render probability cones for the top leans + an HTML dashboard

Outputs land in quant/reports/ (dashboard.html is always the latest) and
quant/journal/ (the accumulating memory of the system).

Run manually:  python daily_run.py [--ensemble]
Schedule:      handled by Windows Task Scheduler (see README in reports/)
"""
from __future__ import annotations

import argparse
import base64
import logging
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("daily_run")

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def _img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", action="store_true", help="use 5-model ensemble (slower)")
    parser.add_argument("--top-cones", type=int, default=3)
    args = parser.parse_args()

    t0 = datetime.now()
    print(f"[{t0:%Y-%m-%d %H:%M}] daily run starting (ensemble={args.ensemble})")

    import screener
    import signal_log
    import forecast

    # 1. Screen
    results = screener.screen(use_ensemble=args.ensemble)
    non_flat = [r for r in results if r.signal != "flat"]
    print(f"  screened {len(results)} symbols · {len(non_flat)} non-flat leans")

    # 2. Journal
    added = signal_log.log_signals(results)
    print(f"  journaled {added} new signals")

    # 3. Score matured
    scored = signal_log.score_matured()
    print(f"  scored {scored} matured signals")

    # 4. Recalibrate
    cal = signal_log.calibrate()
    stats = signal_log.journal_stats()
    print(f"  calibration updated · journal: {stats['total_signals']} total, "
          f"{stats['scored']} scored, live win rate: {stats['live_win_rate']}")

    # 5. Probability cones for top leans by confidence
    top = sorted(non_flat, key=lambda r: r.confidence, reverse=True)[: args.top_cones]
    cone_blocks = []
    for r in top:
        try:
            cone, summary, png = forecast.forecast_ticker(r.ticker, r.signal, r.confidence)
            cone_blocks.append((r, summary, png))
            print(f"  cone {r.ticker}: median day-10 {summary['median_pct']:+.1f}% "
                  f"(90%: {summary['p5_pct']:+.1f}% .. {summary['p95_pct']:+.1f}%)")
        except Exception as e:
            logger.warning("cone for %s failed: %s", r.ticker, e)

    # 6. Dashboard HTML (self-contained, dark)
    rows_html = "".join(
        f"<tr><td>{r.ticker}</td><td class='{r.signal}'>{r.signal.upper()}</td>"
        f"<td>{r.proba*100:.1f}%</td><td>{r.confidence*100:.0f}%</td>"
        f"<td>{signal_log.get_calibrated_win(r.confidence)*100:.1f}%</td>"
        f"<td>${r.last_close:,.2f}</td><td>{r.asset_class}</td></tr>"
        for r in sorted(non_flat, key=lambda x: x.confidence, reverse=True)[:20]
    )
    cal_html = "".join(
        f"<tr><td>{name}</td><td>{b['prior_win']*100:.0f}%</td>"
        f"<td>{(str(round(b['live_win']*100,1))+'% ('+str(b['live_n'])+')') if b['live_n'] else '—'}</td>"
        f"<td><b>{b['posterior_win']*100:.1f}%</b></td></tr>"
        for name, b in cal["bands"].items()
    )
    cones_html = "".join(
        f"<h3>{r.ticker} — {r.signal.upper()} (conf {r.confidence*100:.0f}%) · "
        f"median day-10: {s['median_pct']:+.1f}% · 90% cone {s['p5_pct']:+.1f}%..{s['p95_pct']:+.1f}%</h3>"
        f"<img src='data:image/png;base64,{_img_b64(png)}' style='max-width:100%'>"
        for r, s, png in cone_blocks
    )

    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>dollarsign daily</title>
<style>
 body{{background:#0E1117;color:#E8E6E1;font-family:system-ui,sans-serif;max-width:960px;margin:24px auto;padding:0 16px}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}}
 td,th{{border:1px solid #2A2B38;padding:6px 10px;text-align:left}}
 th{{background:#1A1B24;color:#9B99A1}}
 .long{{color:#00E68A;font-weight:600}} .short{{color:#FF4757;font-weight:600}}
 h1{{font-size:22px}} h2{{font-size:16px;color:#9B99A1;margin-top:28px}} h3{{font-size:13px;color:#9B99A1}}
 .muted{{color:#66656E;font-size:11px}}
</style></head><body>
<h1>dollarsign — daily run · {date.today().isoformat()}</h1>
<p class='muted'>Signals require the model + live-calibrated win rates. Probability cones are Monte Carlo simulations,
not predictions: 90% of simulated futures fall inside the light band IF the calibrated edge is real. Not financial advice.</p>
<h2>Journal / self-improvement state</h2>
<p>Total signals logged: <b>{stats['total_signals']}</b> · scored: <b>{stats['scored']}</b> ·
live win rate: <b>{(str(round(stats['live_win_rate']*100,1))+'%') if stats['live_win_rate'] is not None else 'accumulating…'}</b></p>
<table><tr><th>conf band</th><th>backtest prior</th><th>live (n)</th><th>calibrated win</th></tr>{cal_html}</table>
<h2>Top leans today (top 20 by confidence)</h2>
<table><tr><th>ticker</th><th>lean</th><th>P(up)</th><th>conf</th><th>calibrated win</th><th>price</th><th>class</th></tr>{rows_html}</table>
<h2>Probability cones</h2>
{cones_html}
<p class='muted'>Generated {datetime.now():%Y-%m-%d %H:%M} · runtime {(datetime.now()-t0).seconds}s ·
ensemble={args.ensemble} · journal at quant/journal/signals.csv</p>
</body></html>"""

    out = REPORTS_DIR / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    dated = REPORTS_DIR / f"dashboard_{date.today():%Y%m%d}.html"
    dated.write_text(html, encoding="utf-8")
    print(f"  dashboard: {out}")
    print(f"[{datetime.now():%H:%M}] done in {(datetime.now()-t0).seconds}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
