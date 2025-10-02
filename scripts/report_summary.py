#!/usr/bin/env python3
# scripts/report_summary.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse, yaml
import pandas as pd
import numpy as np
from vix_hedge_states.io import load_all_data
from vix_hedge_states.features import build_features, zscore_frame
from vix_hedge_states.states import fit_states_kmeans, predict_states_kmeans, map_labels_to_names
from vix_hedge_states.backtest import Backtester
from vix_hedge_states.eval import compute_metrics

def _run_policy(cfg, always_on=False):
    df = load_all_data(cfg["paths"]["data_dir"], cfg)
    d = build_features(df)
    use_cols = [c for c in cfg.get("features", {}).get("use_columns", []) if c in d.columns]
    d = zscore_frame(d, use_cols, cfg.get("features", {}).get("rolling_zscore_window", 1260))
    zcols = [c+"_z" for c in use_cols if c+"_z" in d.columns]
    d = d.dropna(subset=zcols).reset_index(drop=True)

    scfg = cfg.get("states", {})
    km = fit_states_kmeans(d, zcols, k=int(scfg.get("k", 4)), random_state=cfg.get("seed", 42))
    raw = predict_states_kmeans(d, zcols, km)
    names = map_labels_to_names(raw, scfg.get("labels", [f"state_{i}" for i in range(int(scfg.get("k", 4)))]))
    d["state"] = names

    if always_on:
        # Hedge in all states & disable triggers
        cfg2 = yaml.safe_load(yaml.dump(cfg))
        cfg2["policy"]["hedge_states"] = list(set(d["state"]))
        cfg2["policy"]["trigger"] = {"type": "none", "rules": {}}
        return Backtester(cfg2).run(d, d["state"])
    else:
        return Backtester(cfg).run(d, d["state"])

def _worst_days(nav, N=20):
    ret = nav.pct_change().dropna()
    idx = ret.nsmallest(N).index
    return ret.loc[idx].sum(), ret.loc[idx]

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--worstN", type=int, default=20)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(PROJECT_ROOT / args.config, "r"))

    timed = _run_policy(cfg, always_on=False)
    base  = _run_policy(cfg, always_on=True)

    outdir = PROJECT_ROOT / "reports"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.append({"period":"Overall","policy":"timed", **compute_metrics(timed["nav"], cfg["evaluation"]["es_alpha"], cfg["evaluation"]["es_alpha_tail"])})
    rows.append({"period":"Overall","policy":"always_on", **compute_metrics(base["nav"], cfg["evaluation"]["es_alpha"], cfg["evaluation"]["es_alpha_tail"])})

    # Subperiods by decade
    decades = [(1990,1999),(2000,2009),(2010,2019),(2020,2029)]
    for a,b in decades:
        a_dt = pd.Timestamp(f"{a}-01-01"); b_dt = pd.Timestamp(f"{b}-12-31")
        t = timed[(timed["date"]>=a_dt)&(timed["date"]<=b_dt)]
        u = base [(base ["date"]>=a_dt)&(base ["date"]<=b_dt)]
        if len(t) > 252 and len(u) > 252:
            rows.append({"period":f"{a}-{b}","policy":"timed", **compute_metrics(t["nav"], cfg["evaluation"]["es_alpha"], cfg["evaluation"]["es_alpha_tail"])})
            rows.append({"period":f"{a}-{b}","policy":"always_on", **compute_metrics(u["nav"], cfg["evaluation"]["es_alpha"], cfg["evaluation"]["es_alpha_tail"])})

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir/"summary_metrics.csv", index=False)

    # Crash capture
    cap_t, _ = _worst_days(timed["nav"], args.worstN)
    cap_b, _ = _worst_days(base ["nav"], args.worstN)
    pd.DataFrame({"policy":["timed","always_on"], "worstN":[args.worstN,args.worstN], "sum_returns":[cap_t,cap_b]}
        ).to_csv(outdir/"crash_capture.csv", index=False)

    print(f"[OK] Wrote {outdir/'summary_metrics.csv'} and {outdir/'crash_capture.csv'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
