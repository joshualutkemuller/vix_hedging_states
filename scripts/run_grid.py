#!/usr/bin/env python3
# scripts/run_grid.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse, yaml
import pandas as pd
import numpy as np
from itertools import product

from vix_hedge_states.io import load_all_data
from vix_hedge_states.features import build_features, zscore_frame
from vix_hedge_states.states import fit_states_kmeans, predict_states_kmeans, map_labels_to_names
from vix_hedge_states.backtest import Backtester
from vix_hedge_states.eval import compute_metrics

def _run_once(cfg, k, tenor, mny, budget, always_on=False):
    C = yaml.safe_load(yaml.dump(cfg))
    C["states"]["k"] = int(k)
    C["policy"]["tenor_days"] = int(tenor)
    C["policy"]["moneyness_pct"] = float(mny)
    C["policy"]["annual_budget"] = float(budget)
    if always_on:
        C["policy"]["hedge_states"] = C["states"]["labels"]
        C["policy"]["trigger"] = {"type":"none", "rules":{}}

    df = load_all_data(C["paths"]["data_dir"], C)
    d  = build_features(df)
    use_cols = [c for c in C.get("features",{}).get("use_columns", []) if c in d.columns]
    d  = zscore_frame(d, use_cols, C.get("features",{}).get("rolling_zscore_window", 1260))
    zc = [c+"_z" for c in use_cols if c+"_z" in d.columns]
    d  = d.dropna(subset=zc).reset_index(drop=True)

    km = fit_states_kmeans(d, zc, k=C["states"]["k"], random_state=C.get("seed",42))
    lab = predict_states_kmeans(d, zc, km)
    d["state"] = map_labels_to_names(lab, C["states"]["labels"])

    bt = Backtester(C).run(d, d["state"])
    m  = compute_metrics(bt["nav"], C["evaluation"]["es_alpha"], C["evaluation"]["es_alpha_tail"])
    m.update({
        "finalNAV": bt["nav"].iloc[-1],
        "spend": bt["spend"].sum(),
        "policy": "always_on" if always_on else "timed",
        "k": k, "tenor": tenor, "moneyness": mny, "budget": budget
    })
    return m

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--k", type=int, nargs="+", default=[3,4,5])
    ap.add_argument("--tenor", type=int, nargs="+", default=[63,126,189])
    ap.add_argument("--moneyness", type=float, nargs="+", default=[-0.05,-0.10,-0.15])
    ap.add_argument("--budget", type=float, nargs="+", default=[0.01,0.02,0.03])
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(PROJECT_ROOT / args.config, "r"))

    rows = []
    for k, t, m, B in product(args.k, args.tenor, args.moneyness, args.budget):
        print(f"[GRID] k={k} tenor={t} mny={m} budget={B} timed")
        rows.append(_run_once(cfg, k, t, m, B, always_on=False))
        print(f"[GRID] k={k} tenor={t} mny={m} budget={B} always_on")
        rows.append(_run_once(cfg, k, t, m, B, always_on=True))

    res = pd.DataFrame(rows)
    # Hedge Efficiency vs always_on
    keyed = ["k","tenor","moneyness","budget"]
    timed = res[res["policy"]=="timed"][keyed+["ES95","ES99","MaxDD","Vol","finalNAV","spend"]].rename(columns=lambda c: c if c in keyed else c+"_t")
    base  = res[res["policy"]=="always_on"][keyed+["ES95","ES99","MaxDD","Vol","finalNAV","spend"]].rename(columns=lambda c: c if c in keyed else c+"_b")
    merged = timed.merge(base, on=keyed, how="inner")
    merged["HE_ES95"] = (merged["ES95_b"] - merged["ES95_t"]) / merged["spend_t"].replace(0,np.nan)
    merged["HE_MaxDD"] = (merged["MaxDD_b"] - merged["MaxDD_t"]) / merged["spend_t"].replace(0,np.nan)

    outdir = PROJECT_ROOT / "reports"
    outdir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(outdir/"grid_results.csv", index=False)
    print(f"[OK] wrote {outdir/'grid_results.csv'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
