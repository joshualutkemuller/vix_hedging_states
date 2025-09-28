#!/usr/bin/env python3
"""
run_states_eda.py
-----------------
Fits K-means states from config, exports:
- reports/states/state_labels.csv           (date, state)
- reports/states/state_frequencies.csv      (counts & pct)
- reports/states/state_dwell_times.csv      (run-lengths per state)
- reports/states/transition_matrix.csv      (empirical transitions)

Figures under reports/figures/states/:
- ts_vix_shaded_states.png
- ts_spy_drawdown_shaded_states.png
- bar_state_frequency.png
- heatmap_state_transitions.png

Usage:
  python -m scripts.run_states_eda --config config/config.yaml
"""

# --- robust imports no matter where you run this ---
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from vix_hedge_states.io import load_all_data
from vix_hedge_states.features import build_features, zscore_frame

# Try to import states helpers from your project. If missing, fall back.
try:
    from vix_hedge_states.states import (
        fit_states_kmeans, predict_states_kmeans, map_labels_to_names
    )
except Exception:
    # Minimal fallback to keep script usable
    from sklearn.cluster import KMeans
    def fit_states_kmeans(df, cols, k=4, random_state=42):
        km = KMeans(n_clusters=k, n_init=25, random_state=random_state)
        km.fit(df[cols].values)
        return km
    def predict_states_kmeans(df, cols, km):
        return km.predict(df[cols].values)
    def map_labels_to_names(labels, names):
        # stable mapping by label index; pad/truncate names as needed
        names = list(names) if names else [f"state_{i}" for i in range(len(set(labels)))]
        lut = {i: names[i % len(names)] for i in sorted(set(labels))}
        return pd.Series(labels).map(lut).values

def save_df(df: pd.DataFrame, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

def plot_and_save(figpath: Path):
    figpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    plt.close()

def compute_transition_matrix(states: pd.Series, state_order=None) -> pd.DataFrame:
    s = states.dropna().astype(str).reset_index(drop=True)
    if state_order is None:
        state_order = sorted(s.unique().tolist())
    idx = {name: i for i, name in enumerate(state_order)}
    n = len(state_order)
    M = np.zeros((n, n), dtype=float)
    for a, b in zip(s[:-1], s[1:]):
        M[idx[a], idx[b]] += 1.0
    row_sums = M.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        P = np.divide(M, row_sums, where=row_sums>0)
    return pd.DataFrame(P, index=state_order, columns=state_order)

def dwell_times(states: pd.Series) -> pd.DataFrame:
    s = states.dropna().astype(str).reset_index(drop=True)
    runs = []
    if s.empty:
        return pd.DataFrame(columns=["state", "length"])
    cur = s.iloc[0]; length = 1
    for val in s.iloc[1:]:
        if val == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur, length = val, 1
    runs.append((cur, length))
    dt = pd.DataFrame(runs, columns=["state","length"])
    return dt

def shade_states(ax, dates: pd.Series, states: pd.Series, palette=None, alpha=0.12):
    s = states.astype(str).reset_index(drop=True)
    d = pd.to_datetime(dates).reset_index(drop=True)
    uniq = list(pd.Series(s).dropna().unique())
    if palette is None:
        # simple repeating palette
        palette = {
            name: clr for name, clr in zip(
                uniq,
                ["#1f77b4","#2ca02c","#ff7f0e","#d62728","#9467bd","#8c564b"]
            )
        }
    # draw spans per run
    cur = s.iloc[0]; start = d.iloc[0]
    for i in range(1, len(s)):
        if s.iloc[i] != cur:
            ax.axvspan(start, d.iloc[i], color=palette.get(cur, "#999999"), alpha=alpha, lw=0)
            cur = s.iloc[i]
            start = d.iloc[i]
    ax.axvspan(start, d.iloc[-1], color=palette.get(cur, "#999999"), alpha=alpha, lw=0)
    # legend swatches
    handles = [plt.Line2D([0],[0], color=palette[k], lw=6) for k in uniq]
    ax.legend(handles, uniq, title="State", fontsize=8, loc="upper left")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    with open(PROJECT_ROOT / args.config, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    states_cfg = cfg.get("states", {})
    k = int(states_cfg.get("k", 4))
    labels_cfg = states_cfg.get("labels", [f"state_{i}" for i in range(k)])

    # Load data & features
    df = load_all_data(paths["data_dir"], cfg)
    d = build_features(df)

    # Z-scores for configured features
    fcfg = cfg.get("features", {})
    use_cols = [c for c in fcfg.get("use_columns", []) if c in d.columns]
    d = zscore_frame(d, use_cols, fcfg.get("rolling_zscore_window", 1260))
    zcols = [c+"_z" for c in use_cols if c+"_z" in d.columns]
    d = d.dropna(subset=zcols).reset_index(drop=True)

    # Fit & predict K-means states
    km = fit_states_kmeans(d, zcols, k=k, random_state=cfg.get("seed", 42))
    raw_labels = predict_states_kmeans(d, zcols, km)
    named = map_labels_to_names(raw_labels, labels_cfg)
    d["state"] = named

    # --- Outputs (tables)
    states_dir = PROJECT_ROOT / "reports" / "states"
    print(states_dir)
    figs_dir = PROJECT_ROOT / "reports" / "figures" / "states"
    save_df(d[["date","state"]], states_dir / "state_labels.csv")

    # Frequency table
    freq = (d["state"].value_counts().rename_axis("state").to_frame("count")
            .assign(pct=lambda x: x["count"]/x["count"].sum()))
    save_df(freq.reset_index(), states_dir / "state_frequencies.csv")

    # Dwell times (run-lengths)
    dwell = dwell_times(d["state"])
    save_df(dwell, states_dir / "state_dwell_times.csv")

    # Transition matrix
    order = list(freq.index)  # order by prevalence
    P = compute_transition_matrix(d["state"], state_order=order)
    P.to_csv(states_dir / "transition_matrix.csv", index=True)

    # --- Plots
    # 1) VIX with shaded states
    if "vix" in d.columns:
        plt.figure()
        plt.plot(d["date"], d["vix"])
        ax = plt.gca()
        shade_states(ax, d["date"], d["state"])
        ax.set_title("VIX with Regime Shading")
        ax.set_xlabel("Date"); ax.set_ylabel("VIX")
        plot_and_save(figs_dir / "ts_vix_shaded_states.png")

    # 2) SPY drawdown with shaded states
    if "spy_tr" in d.columns:
        level = pd.to_numeric(d["spy_tr"], errors="coerce")
        peak = level.cummax()
        dd = level/peak - 1.0
        plt.figure()
        plt.plot(d["date"], dd)
        ax = plt.gca()
        shade_states(ax, d["date"], d["state"])
        ax.set_title("SPY Drawdown with Regime Shading")
        ax.set_xlabel("Date"); ax.set_ylabel("Drawdown")
        plot_and_save(figs_dir / "ts_spy_drawdown_shaded_states.png")

    # 3) Bar: state frequency
    plt.figure()
    freq_plot = freq.sort_values("count", ascending=False)
    plt.bar(freq_plot.index, freq_plot["count"])
    plt.title("State Frequency (counts)")
    plt.xlabel("State"); plt.ylabel("Count")
    plt.xticks(rotation=20)
    plot_and_save(figs_dir / "bar_state_frequency.png")

main()