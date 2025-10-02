#!/usr/bin/env python3
# scripts/run_backtest.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse, yaml
import pandas as pd
from vix_hedge_states.io import load_all_data
from vix_hedge_states.features import build_features, zscore_frame
from vix_hedge_states.states import fit_states_kmeans, predict_states_kmeans, map_labels_to_names
from vix_hedge_states.backtest import Backtester

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args(argv)

    with open(PROJECT_ROOT / args.config, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    fcfg = cfg.get("features", {})
    scfg = cfg.get("states", {})

    df = load_all_data(paths["data_dir"], cfg)
    d = build_features(df)

    use_cols = [c for c in fcfg.get("use_columns", []) if c in d.columns]
    d = zscore_frame(d, use_cols, fcfg.get("rolling_zscore_window", 1260))
    zcols = [c+"_z" for c in use_cols if c+"_z" in d.columns]
    d = d.dropna(subset=zcols).reset_index(drop=True)

    # Fit states
    km = fit_states_kmeans(d, zcols, k=int(scfg.get("k", 4)), random_state=cfg.get("seed", 42))
    labels = predict_states_kmeans(d, zcols, km)
    state_names = map_labels_to_names(labels, scfg.get("labels", [f"state_{i}" for i in range(int(scfg.get("k", 4)))]))
    d["state"] = state_names

    # Run backtest
    bt = Backtester(cfg).run(d, d["state"])

    # Save outputs
    outdir = PROJECT_ROOT / "reports" / "backtest"
    outdir.mkdir(parents=True, exist_ok=True)
    bt.to_csv(outdir / "backtest_nav.csv", index=False)

    print(f"[OK] Backtest done. Saved -> {outdir/'backtest_nav.csv'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
