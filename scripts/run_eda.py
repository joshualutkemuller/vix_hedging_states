#!/usr/bin/env python3
"""
run_eda.py
----------
Basic EDA for the VIX term-structure hedging project.

Outputs to reports/eda/:
- eda_coverage.csv            : date range, row counts
- eda_missingness.csv         : missing value counts & percentages
- eda_descriptive_stats.csv   : describe() for key columns
- eda_correlations.csv        : Pearson correlation matrix (key numeric cols)

Figures to reports/figures/eda/:
- ts_vix_vx.png               : time series of VIX, VX1-4
- ts_slopes_curvature.png     : time series of slope12, slope13, curvature
- hist_vix.png                : histogram of VIX
- hist_slopes.png             : histograms of slope12 & slope13
- corr_heatmap.png            : correlation heatmap of key columns

Usage:
  python -m scripts.run_eda --config config/config.yaml
"""

import sys
from pathlib import Path

# Make src/ importable no matter where you run the script from
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # insert(0) > append

import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vix_hedge_states.io import load_all_data
from vix_hedge_states.features import build_features, zscore_frame
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def save_df(df: pd.DataFrame, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

def plot_and_save(figpath: Path):
    figpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # extract columns to use
    fcfg = cfg.get("features", {})
    use_cols = fcfg.get("use_columns", True)
    print(use_cols)
    # Load & engineer features (same path as the rest of the project)
    df = load_all_data(cfg["paths"]["data_dir"], cfg)
    d = build_features(df)

    # Add z-scores for configured columns (if present)
    use_cols = cfg["features"]["use_columns"]
    d = zscore_frame(d, [c for c in use_cols if c in d.columns], cfg["features"]["rolling_zscore_window"])

    # Ensure date sorted
    d = d.sort_values("date").reset_index(drop=True)

    # -------------------- Tabular EDA --------------------
    reports_dir = PROJECT_ROOT / "reports" / "eda"
    figures_dir = PROJECT_ROOT / "reports" / "figures" / "eda"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Coverage summary
    coverage = pd.DataFrame({
        "first_date": [d["date"].min()],
        "last_date": [d["date"].max()],
        "n_rows": [len(d)],
        "n_days": [(d["date"].max() - d["date"].min()).days if len(d) > 0 else 0]
    })
    save_df(coverage, reports_dir / "eda_coverage.csv")

    # Missingness
    miss = d.isna().sum().rename("n_missing").to_frame()
    miss["pct_missing"] = miss["n_missing"] / len(d) * 100.0
    miss = miss.reset_index().rename(columns={"index": "column"})
    save_df(miss, reports_dir / "eda_missingness.csv")

    # Numeric columns (reasonable default set)
    numeric_cols = [c for c in use_cols if c in d.columns]

    # Descriptive stats
    desc = d[numeric_cols].describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]).T.reset_index().rename(columns={"index":"feature"})
    save_df(desc, reports_dir / "eda_descriptive_stats.csv")

    # Correlations
    corr = d[numeric_cols].corr(method="pearson")
    corr_out = corr.reset_index().rename(columns={"index":"feature"})
    save_df(corr_out, reports_dir / "eda_correlations.csv")

    # -------------------- Plots --------------------
    # 1) Time series: VIX & VX1-4
    plt.figure()
    if "vix" in d: plt.plot(d["date"], d["vix"], label="VIX")
    for c in ["vx1","vx2","vx3","vx4"]:
        if c in d: plt.plot(d["date"], d[c], label=c.upper())
    plt.title("VIX and VIX Futures (VX1–VX4)")
    plt.xlabel("Date"); plt.ylabel("Level")
    plt.legend()
    plot_and_save(figures_dir / "ts_vix_vx.png")

    # 2) Time series: slopes & curvature
    plt.figure()
    if "slope12" in d: plt.plot(d["date"], d["slope12"], label="slope12 (vx2-vx1)")
    if "slope13" in d: plt.plot(d["date"], d["slope13"], label="slope13 (vx3-vx1)")
    if "curvature" in d: plt.plot(d["date"], d["curvature"], label="curvature (vx2 - 0.5*(vx1+vx3))")
    plt.title("Term-Structure Slopes & Curvature")
    plt.xlabel("Date"); plt.ylabel("Spread (pts)")
    plt.legend()
    plot_and_save(figures_dir / "ts_slopes_curvature.png")

    # 3) Histogram: VIX
    if "vix" in d:
        plt.figure()
        d["vix"].dropna().plot(kind="hist", bins=60)
        plt.title("Distribution of VIX")
        plt.xlabel("VIX level"); plt.ylabel("Frequency")
        plot_and_save(figures_dir / "hist_vix.png")

    # 4) Histograms: slope12 & slope13
    plt.figure()
    plotted_any = False
    if "slope12" in d:
        d["slope12"].dropna().plot(kind="hist", bins=60, alpha=0.6)
        plotted_any = True
    if "slope13" in d:
        d["slope13"].dropna().plot(kind="hist", bins=60, alpha=0.6)
        plotted_any = True
    if plotted_any:
        plt.title("Distribution of Slopes (slope12, slope13)")
        plt.xlabel("Spread (pts)"); plt.ylabel("Frequency")
        plot_and_save(figures_dir / "hist_slopes.png")
    else:
        plt.close()

    # 5) Correlation heatmap (matplotlib only)
    if len(numeric_cols) >= 2:
        plt.figure()
        im = plt.imshow(corr.values, origin="lower", aspect="auto")
        plt.colorbar(im)
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
        plt.yticks(range(len(numeric_cols)), numeric_cols)
        plt.title("Correlation Heatmap (Pearson)")
        plot_and_save(figures_dir / "corr_heatmap.png")

    # 6) Forward return relation (optional & lightweight)
    #    Example: forward 20D SPY TR vs. slope12 (does term carry predict risk?)
    if "spy_tr" in d and "slope12" in d:
        d = d.copy()
        d["ret_20d_fwd"] = d["spy_tr"].pct_change(periods=20).shift(-20)
        mask = d[["slope12","ret_20d_fwd"]].dropna()
        if len(mask) > 100:
            plt.figure()
            plt.scatter(mask["slope12"], mask["ret_20d_fwd"], s=6)
            plt.title("Forward 20D SPY Return vs. slope12")
            plt.xlabel("slope12 (vx2 - vx1)"); plt.ylabel("SPY TR 20D forward return")
            plot_and_save(figures_dir / "scatter_slope12_vs_fwd20d.png")

    # --- 7) SPY total return level
    if "spy_tr" in d:
        plt.figure()
        plt.plot(d["date"], d["spy_tr"])
        plt.title("SPY Total-Return Index")
        plt.xlabel("Date"); plt.ylabel("Index level")
        plot_and_save(figures_dir / "ts_spy_tr.png")

    # --- 8) SPY daily returns (log) and daily total returns
    if "ret" in d:
        plt.figure()
        d["ret"].dropna().plot(kind="hist", bins=80)
        plt.title("SPY Daily Total Log Returns")
        plt.xlabel("log total return"); plt.ylabel("Frequency")
        plot_and_save(figures_dir / "hist_spy_total_returns_log.png")

        plt.figure()
        d["spy_tr"].dropna().plot(kind="hist", bins=80)
        plt.title("SPY Daily Total Returns")
        plt.xlabel("total return"); plt.ylabel("Frequency")
        plot_and_save(figures_dir / "hist_spy_total_returns.png")


    # --- 9) SPY rolling realized vol (20D & 60D, annualized)
    if "ret" in d:
        rv20 = d["ret"].rolling(21).std()*np.sqrt(252)*100
        rv60 = d["ret"].rolling(63).std()*np.sqrt(252)*100
        plt.figure()
        plt.plot(d["date"], rv20, label="RV20 (%, ann.)")
        plt.plot(d["date"], rv60, label="RV60 (%, ann.)")
        plt.title("SPY Rolling Realized Volatility")
        plt.xlabel("Date"); plt.ylabel("Vol (%)")
        plt.legend()
        plot_and_save(figures_dir / "ts_spy_rolling_vol.png")

    # --- 10) SPY drawdown curve
    if "spy_tr" in d:
        level = d["spy_tr"].astype(float)
        peak = level.cummax()
        dd = level/peak - 1.0
        plt.figure()
        plt.plot(d["date"], dd)
        plt.title("SPY Drawdown")
        plt.xlabel("Date"); plt.ylabel("Drawdown")
        plot_and_save(figures_dir / "ts_spy_drawdown.png")

    # --- 11) SPY vs VIX (twin axes)
    if "spy_tr" in d and "vix" in d:
        fig, ax1 = plt.subplots()
        ax1.plot(d["date"], d["spy_tr"], label="SPY TR")
        ax1.set_xlabel("Date"); ax1.set_ylabel("SPY TR")
        ax2 = ax1.twinx()
        ax2.plot(d["date"], d["vix"], alpha=0.6, label="VIX")
        ax2.set_ylabel("VIX")
        ax1.set_title("SPY vs VIX")
        plot_and_save(figures_dir / "ts_spy_vs_vix.png")

    # --- 12) Scatter: SPY daily returns vs VIX level
    if "ret" in d and "vix" in d:
        mask = d[["ret","vix"]].dropna()
        if len(mask) > 200:
            plt.figure()
            plt.scatter(mask["vix"], mask["ret"], s=6)
            plt.title("SPY Daily Log Return vs VIX")
            plt.xlabel("VIX level"); plt.ylabel("log return")
            plot_and_save(figures_dir / "scatter_ret_vs_vix.png")

    # --- 13) Scatter: forward 20D SPY return vs slope12 (term structure)
    if "spy_tr" in d and "slope12" in d:
        d = d.copy()
        d["ret_20d_fwd"] = d["spy_tr"].pct_change(20).shift(-20)
        mask = d[["slope12","ret_20d_fwd"]].dropna()
        if len(mask) > 200:
            plt.figure()
            plt.scatter(mask["slope12"], mask["ret_20d_fwd"], s=6)
            plt.title("Forward 20D SPY Return vs slope12 (vx2 - vx1)")
            plt.xlabel("slope12 (pts)"); plt.ylabel("forward 20D return")
            plot_and_save(figures_dir / "scatter_slope12_vs_fwd20d.png")
            
    # ---- 14) SPY rolling percent-change charts (1W, 2W, 1M, 3M) ----
    # Windows (trading days): 1W≈5, 2W≈10, 1M≈21, 3M≈63
    if "spy_tr" in d:
            import matplotlib.ticker as mtick

            def _annotate_min_max_latest(ax, dates, series):
                s = series.dropna()
                if s.empty:
                    return
                i_min, i_max, i_last = s.idxmin(), s.idxmax(), s.index[-1]
                pts = [("Min", i_min), ("Max", i_max), ("Latest", i_last)]
                used = set()
                for tag, idx in pts:
                    if idx in used: 
                        continue
                    used.add(idx)
                    x, y = dates.iloc[idx], series.iloc[idx]
                    ax.scatter([x], [y], s=30)
                    dy = (series.std(skipna=True) or 0.0) * (0.15 if tag != "Max" else -0.15)
                    ax.annotate(
                        f"{tag}\n{pd.to_datetime(x).date()}\n{y:.2%}",
                        xy=(x, y), xytext=(x, y + dy),
                        textcoords="data", arrowprops=dict(arrowstyle="->", lw=1),
                        fontsize=9, ha="center", va="bottom" if dy >= 0 else "top",
                    )

            def _label_lines_right(ax, dates, items, x_pos=0.92):
                """Inline labels near the right edge for each horizontal line."""
                x0, x1 = dates.iloc[0], dates.iloc[-1]
                x_ann = x0 + (x1 - x0) * x_pos
                for y, label in items:
                    ax.annotate(label, xy=(x_ann, y), xycoords="data",
                                xytext=(5, 0), textcoords="offset points",
                                fontsize=8, va="center", ha="left")

            def plot_rolling_change(days: int, label: str, fname: str):
                series = d["spy_tr"].astype(float).pct_change(periods=days)
                mu = series.mean(skipna=True)
                sd = series.std(skipna=True)

                fig, ax = plt.subplots()
                ax.plot(d["date"], series, label=f"{label} change")
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
                ax.set_title(f"SPY {label} Change (%): level-over-level")
                ax.set_xlabel("Date"); ax.set_ylabel("Percent")

                # Reference lines with labels
                ref_lines = [
                    (0.0, "-",  "Zero (0%)"),
                    (mu,  ":",  f"Mean ({mu:.2%})"),
                    (mu + sd,  "--", f"Mean +1σ ({(mu+sd):.2%})"),
                    (mu - sd,  "--", f"Mean -1σ ({(mu-sd):.2%})"),
                    (mu + 2*sd,"-.", f"Mean +2σ ({(mu+2*sd):.2%})"),
                    (mu - 2*sd,"-.", f"Mean -2σ ({(mu-2*sd):.2%})"),
                ]
                legend_items = []
                for y, style, text in ref_lines:
                    h = ax.axhline(y=y, linestyle=style, linewidth=1, label=text)
                    legend_items.append(h)

                # Inline labels near right edge (in addition to legend)
                _label_lines_right(ax, d["date"], [(y, txt) for y, _, txt in ref_lines])

                # Min / Max / Latest annotations
                _annotate_min_max_latest(ax, d["date"], series)

                # Legend
                ax.legend(title="Reference lines", fontsize=8)
                plot_and_save(figures_dir / fname)

    # Windows: 1W≈5d, 2W≈10d, 1M≈21d, 3M≈63d
    plot_rolling_change(5,   "1-Week (~5d)",  "ts_spy_change_1W.png")
    plot_rolling_change(10,  "2-Week (~10d)", "ts_spy_change_2W.png")
    plot_rolling_change(21,  "1-Month (~21d)","ts_spy_change_1M.png")
    plot_rolling_change(63,  "3-Month (~63d)","ts_spy_change_3M.png")

    # ---- 15) VIX rolling % change with labeled reference lines + min/max/latest annotations ----
    if "vix" in d:
        import matplotlib.ticker as mtick

        def _annotate_min_max_latest(ax, dates, series):
            s = series.dropna()
            if s.empty:
                return
            i_min, i_max, i_last = s.idxmin(), s.idxmax(), s.index[-1]
            pts = [("Min", i_min), ("Max", i_max), ("Latest", i_last)]
            used = set()
            for tag, idx in pts:
                if idx in used: 
                    continue
                used.add(idx)
                x, y = dates.iloc[idx], series.iloc[idx]
                ax.scatter([x], [y], s=30)
                dy = (series.std(skipna=True) or 0.0) * (0.15 if tag != "Max" else -0.15)
                ax.annotate(
                    f"{tag}\n{pd.to_datetime(x).date()}\n{y:.2%}",
                    xy=(x, y), xytext=(x, y + dy),
                    textcoords="data", arrowprops=dict(arrowstyle="->", lw=1),
                    fontsize=9, ha="center", va="bottom" if dy >= 0 else "top",
                )

        def _label_lines_right(ax, dates, items, x_pos=0.92):
            """Inline labels near the right edge for each horizontal line."""
            x0, x1 = dates.iloc[0], dates.iloc[-1]
            x_ann = x0 + (x1 - x0) * x_pos
            for y, label in items:
                ax.annotate(label, xy=(x_ann, y), xycoords="data",
                            xytext=(5, 0), textcoords="offset points",
                            fontsize=8, va="center", ha="left")

        def plot_rolling_change(days: int, label: str, fname: str):
            series = d["vix"].astype(float).pct_change(periods=days)
            mu = series.mean(skipna=True)
            sd = series.std(skipna=True)

            fig, ax = plt.subplots()
            ax.plot(d["date"], series, label=f"{label} change")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            ax.set_title(f"VIX {label} Change (%): level-over-level")
            ax.set_xlabel("Date"); ax.set_ylabel("Percent")

            # Reference lines with labels
            ref_lines = [
                (0.0, "-",  "Zero (0%)"),
                (mu,  ":",  f"Mean ({mu:.2%})"),
                (mu + sd,  "--", f"Mean +1σ ({(mu+sd):.2%})"),
                (mu - sd,  "--", f"Mean -1σ ({(mu-sd):.2%})"),
                (mu + 2*sd,"-.", f"Mean +2σ ({(mu+2*sd):.2%})"),
                (mu - 2*sd,"-.", f"Mean -2σ ({(mu-2*sd):.2%})"),
            ]
            legend_items = []
            for y, style, text in ref_lines:
                h = ax.axhline(y=y, linestyle=style, linewidth=1, label=text)
                legend_items.append(h)

            # Inline labels near right edge (in addition to legend)
            _label_lines_right(ax, d["date"], [(y, txt) for y, _, txt in ref_lines])

            # Min / Max / Latest annotations
            _annotate_min_max_latest(ax, d["date"], series)

            # Legend
            ax.legend(title="Reference lines", fontsize=8)
            plot_and_save(figures_dir / fname)

    # Windows: 1W≈5d, 2W≈10d, 1M≈21d, 3M≈63d
    plot_rolling_change(5,   "1-Week (~5d)",  "ts_vix_change_1W.png")
    plot_rolling_change(10,  "2-Week (~10d)", "ts_vix_change_2W.png")
    plot_rolling_change(21,  "1-Month (~21d)","ts_vix_change_1M.png")
    plot_rolling_change(63,  "3-Month (~63d)","ts_vix_change_3M.png")
    
    # ---- 16) SPY Volume EDA (optional if spy_ohlcv.csv exists) ----
    from matplotlib.ticker import FuncFormatter

    def _fmt_millions(x, pos):
        try:
            return f"{x/1_000_000:.0f}M"
        except Exception:
            return str(x)

    spy_ohlcv_path = (PROJECT_ROOT / cfg["paths"]["data_dir"] / "spy_ohlcv.csv")
    if spy_ohlcv_path.exists():
        spy_ohlcv = pd.read_csv(spy_ohlcv_path)

        # Standardize column names (handle variations like 'volume', 'Adj Close', etc.)
        cols_map = {c: c.strip() for c in spy_ohlcv.columns}
        spy_ohlcv.rename(columns=cols_map, inplace=True)

        # Ensure date is datetime
        if "date" not in spy_ohlcv.columns:
            # yfinance may export 'Date'
            if "Date" in spy_ohlcv.columns:
                spy_ohlcv.rename(columns={"Date": "date"}, inplace=True)
            else:
                raise ValueError("spy_ohlcv.csv missing a 'date' (or 'Date') column.")
        spy_ohlcv["date"] = pd.to_datetime(spy_ohlcv["date"])

        # Find price/volume columns
        price_col = "Close" if "Close" in spy_ohlcv.columns else ("Adj Close" if "Adj Close" in spy_ohlcv.columns else None)
        vol_col = "Volume" if "Volume" in spy_ohlcv.columns else ("volume" if "volume" in spy_ohlcv.columns else None)
        if price_col is None or vol_col is None:
            print("[WARN] spy_ohlcv.csv missing Close/Adj Close or Volume; skipping volume EDA.")
        else:
            dfv = spy_ohlcv[["date", price_col, vol_col]].rename(columns={price_col: "close", vol_col: "volume"}).copy()

            # ---- CLEAN NUMERICS: remove commas/whitespace, coerce to float ----
            for c in ["close", "volume"]:
                dfv[c] = (
                    dfv[c]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                )
                dfv[c] = pd.to_numeric(dfv[c], errors="coerce")

            # Drop rows with no volume/price
            dfv = dfv.dropna(subset=["close", "volume"]).sort_values("date").reset_index(drop=True)

            # Rolling averages
            dfv["vol_ma20"] = dfv["volume"].rolling(20).mean()
            dfv["vol_ma60"] = dfv["volume"].rolling(60).mean()

            # Volume z-score on a long window; guard against zero std
            roll_mu = dfv["volume"].rolling(252, min_periods=60).mean()
            roll_sd = dfv["volume"].rolling(252, min_periods=60).std(ddof=0)
            roll_sd = roll_sd.replace(0, np.nan)  # avoid division by zero
            dfv["vol_z"] = (dfv["volume"] - roll_mu) / roll_sd

            # Dollar volume
            dfv["dollar_vol"] = dfv["close"] * dfv["volume"]

            # 1) Volume with 20D/60D averages
            plt.figure()
            plt.plot(dfv["date"], dfv["volume"], label="Volume")
            plt.plot(dfv["date"], dfv["vol_ma20"], label="20D MA")
            plt.plot(dfv["date"], dfv["vol_ma60"], label="60D MA")
            ax = plt.gca()
            ax.yaxis.set_major_formatter(FuncFormatter(_fmt_millions))
            plt.title("SPY Volume (with 20D/60D MAs)")
            plt.xlabel("Date"); plt.ylabel("Shares")
            plt.legend()
            plot_and_save(figures_dir / "ts_spy_volume_ma.png")

            # 2) Volume z-score with ±1σ / ±2σ lines (z already standardized)
            plt.figure()
            plt.plot(dfv["date"], dfv["vol_z"], label="Volume z-score")
            plt.axhline(0, linestyle="-", linewidth=1, label="Mean (0)")
            plt.axhline( 1, linestyle=":", linewidth=1, label="+1σ")
            plt.axhline(-1, linestyle=":", linewidth=1, label="-1σ")
            plt.axhline( 2, linestyle="--", linewidth=1, label="+2σ")
            plt.axhline(-2, linestyle="--", linewidth=1, label="-2σ")
            plt.title("SPY Volume Z-Score (252D lookback)")
            plt.xlabel("Date"); plt.ylabel("Z-score")
            plt.legend()
            plot_and_save(figures_dir / "ts_spy_volume_z.png")

            # 3) Dollar volume
            plt.figure()
            plt.plot(dfv["date"], dfv["dollar_vol"], label="Dollar Volume")
            ax = plt.gca()
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"${x/1_000_000_000:.1f}B"))
            plt.title("SPY Dollar Volume (Close × Shares)")
            plt.xlabel("Date"); plt.ylabel("USD")
            plt.legend()
            plot_and_save(figures_dir / "ts_spy_dollar_volume.png")

            # 4) |Return| vs Volume z
            if "ret" in d.columns:
                m = d[["date", "ret"]].merge(dfv[["date", "vol_z"]], on="date", how="inner").dropna()
                if len(m) > 200:
                    plt.figure()
                    plt.scatter(np.abs(m["ret"]), m["vol_z"], s=6)
                    plt.title("Abs(SPY Daily Log Return) vs Volume Z-Score")
                    plt.xlabel("|log return|"); plt.ylabel("Volume z-score")
                    plot_and_save(figures_dir / "scatter_absret_vs_volz.png")

            # 5) RV20 vs Volume z (twin axes)
            if "rv20" in d.columns:
                m2 = d[["date", "rv20"]].merge(dfv[["date", "vol_z"]], on="date", how="inner").dropna()
                if len(m2) > 60:
                    fig, ax1 = plt.subplots()
                    ax1.plot(m2["date"], m2["rv20"], label="RV20 (%, ann.)")
                    ax1.set_xlabel("Date"); ax1.set_ylabel("RV20 (%)")
                    ax2 = ax1.twinx()
                    ax2.plot(m2["date"], m2["vol_z"], label="Volume z", alpha=0.7)
                    ax2.set_ylabel("Volume z-score")
                    ax1.set_title("SPY Realized Vol (RV20) vs Volume z-score")
                    plot_and_save(figures_dir / "ts_spy_rv20_vs_volz.png")
    else:
        print(f"[INFO] Skipping volume EDA; missing {spy_ohlcv_path}")


    # Write a tiny markdown summary
    md = PROJECT_ROOT / "reports" / "eda" / "eda_summary.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(
        "# EDA Summary\n\n"
        "- Coverage, missingness, descriptive stats, and correlations saved as CSVs in this folder.\n"
        "- Time series and distributions saved to `reports/figures/eda/`.\n"
        "- Optional scatter shows relation between term slope and forward SPY returns.\n"
        "\nRe-run this script after updating data or config to refresh outputs.\n"
    )

    print(f"[OK] EDA complete. Tables -> {reports_dir}, figures -> {figures_dir}")

if __name__ == "__main__":
    main()
