"""Test if a model ensemble beats any single model on equities-pooled.

Approach: run each of 6 models walk-forward on the equities pool,
collect their P(up) outputs per (symbol, date), then test 3 ensemble flavors:
  - AVG: simple mean of all 6 model probabilities
  - AVG_TOP3: mean of top 3 by individual Sharpe (LightGBM, XGB, HistGBM)
  - MEDIAN: median of all 6 (robust to outlier model)
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import indicators
import regime
import backtest
import ml
import models
import class_xgb
from data import sources

logging.basicConfig(level=logging.WARNING)
np.random.seed(42)


def main() -> int:
    start = (date.today() - timedelta(days=4 * 365)).isoformat()

    print("\n" + "=" * 72)
    print("  ENSEMBLE TEST — equities-pooled, 4y, all 6 models + 3 ensembles")
    print("=" * 72)

    symbols = class_xgb.CLASSES["equities"]
    symbol_dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = sources.get_bars(sym, start, use_cache=True)
            df = indicators.attach_all(df)
            df = regime.label(df)
            symbol_dfs[sym] = df
        except Exception as e:
            print(f"  skip {sym}: {e}")

    pooled = class_xgb.build_pooled_features(symbol_dfs, horizon=10).copy()
    pooled["date"] = pd.to_datetime(pooled["date"], utc=True)
    pooled["forward_date"] = pd.to_datetime(pooled["forward_date"], utc=True)

    feat_cols = list(ml.FEATURE_COLS)
    all_dates = sorted(pooled["date"].dropna().unique())
    n_dates = len(all_dates)
    train_size, step_size = 750, 60

    # For each model, collect (model, symbol) -> probability series
    proba_by_model: dict[str, dict[str, pd.Series]] = {}

    for model_name, fit_fn in models.MODELS.items():
        t0 = time.time()
        sym_probas: dict[str, pd.Series] = {}
        for sym in symbol_dfs:
            sym_dates = pd.DatetimeIndex(sorted(pooled.loc[pooled["symbol"] == sym, "date"].unique()))
            sym_probas[sym] = pd.Series(np.nan, index=sym_dates, dtype=float)

        i = train_size
        while i < n_dates:
            cutoff = all_dates[i]
            end_idx = min(i + step_size, n_dates)
            test_dates = set(all_dates[i:end_idx])

            train_mask = (
                (pooled["forward_date"] < cutoff)
                & pooled["target"].notna()
                & pooled["forward_date"].notna()
            )
            train = pooled[train_mask].dropna(subset=feat_cols)
            if len(train) < 200:
                i += step_size
                continue

            try:
                X = train[feat_cols].astype(float).to_numpy()
                y = train["target"].astype(int).to_numpy()
                if len(np.unique(y)) < 2:
                    i += step_size
                    continue
                model = fit_fn(X, y)
                test = pooled[pooled["date"].isin(test_dates)].dropna(subset=feat_cols)
                if not test.empty:
                    proba = model.predict_proba(test[feat_cols].astype(float).to_numpy())[:, 1]
                    for ridx, p in zip(test.index, proba):
                        row = test.loc[ridx]
                        sym_probas[row["symbol"]].loc[row["date"]] = p
            except Exception as e:
                logging.debug("step failed: %s", e)
            i += step_size

        proba_by_model[model_name] = sym_probas
        print(f"  {model_name:<14} fit_seconds={time.time() - t0:.1f}")

    # ─── Build ensembles ───
    # Stack per-symbol probabilities across models into a (date x model) matrix
    # then apply aggregation function. Skip rows where >half of models are NaN.
    def ensemble_per_symbol(agg: str, model_subset: list[str]) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for sym in symbol_dfs:
            cols = {m: proba_by_model[m][sym] for m in model_subset}
            mat = pd.DataFrame(cols)
            # Need at least half the models to have a prediction
            min_n = max(1, len(model_subset) // 2)
            valid = mat.notna().sum(axis=1) >= min_n
            if agg == "mean":
                ens = mat.mean(axis=1)
            elif agg == "median":
                ens = mat.median(axis=1)
            else:
                raise ValueError(agg)
            ens.loc[~valid] = np.nan
            out[sym] = ens
        return out

    ensembles = {
        "ENS_AVG_all":   ensemble_per_symbol("mean", list(models.MODELS.keys())),
        "ENS_AVG_top3":  ensemble_per_symbol("mean", ["lightgbm", "xgboost", "hist_gbm"]),
        "ENS_MEDIAN_all": ensemble_per_symbol("median", list(models.MODELS.keys())),
    }

    # ─── Backtest single models + ensembles, gather per-symbol metrics ───
    def backtest_set(name: str, sym_probas: dict[str, pd.Series]) -> dict:
        per_sym = {}
        for sym, df in symbol_dfs.items():
            proba = sym_probas[sym]
            proba_aligned = proba.reindex(df.index)
            positions = pd.Series(0.0, index=df.index)
            positions[proba_aligned > 0.55] = 1.0
            positions[proba_aligned < 0.45] = -1.0
            try:
                m = backtest.run(df, positions, fee_bps=1.0, slippage_bps=1.0).metrics
                per_sym[sym] = m
            except Exception as e:
                pass

        wins = [m["win_rate"] for m in per_sym.values()]
        rets = [m["total_return"] for m in per_sym.values()]
        shs = [m["sharpe"] for m in per_sym.values()]
        return {
            "med_win": np.median(wins) * 100 if wins else 0,
            "med_ret": np.median(rets) * 100 if rets else 0,
            "med_sh": np.median(shs) if shs else 0,
            "n_profit": sum(1 for r in rets if r > 0),
            "n_total": len(rets),
        }

    print("\n  COMPARISON (median across 7 equities, ranked by Sharpe):")
    print(f"  {'model':<18} {'med_win':>8} {'med_ret':>8} {'med_sh':>7} {'%profit':>9}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*7} {'-'*9}")

    rows = []
    for name in models.MODELS:
        rows.append((name, backtest_set(name, proba_by_model[name])))
    for name, sym_probas in ensembles.items():
        rows.append((name, backtest_set(name, sym_probas)))

    rows.sort(key=lambda x: x[1]["med_sh"], reverse=True)
    for name, m in rows:
        is_ens = name.startswith("ENS_")
        marker = "*" if is_ens else " "
        print(
            f" {marker}{name:<18} {m['med_win']:>7.1f}% {m['med_ret']:>+7.1f}% "
            f"{m['med_sh']:>+6.2f}  {m['n_profit']}/{m['n_total']}"
        )
    print("\n  * = ensemble of multiple models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
