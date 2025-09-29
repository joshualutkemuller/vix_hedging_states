# src/vix_hedge_states/plots.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter

__all__ = [
    "plot_and_save",
    "fmt_percent_axis",
    "fmt_billions",
    "fmt_millions",
    "shade_states",
    "annotate_min_max_latest",
    "add_hlines_with_labels",
    "ts_with_bands",
    "heatmap",
]

# ---------- basic IO ----------

def plot_and_save(path: Path, dpi: int = 150):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

# ---------- formatters ----------

def fmt_percent_axis(ax, xmax: float = 1.0):
    """Format y-axis as percent (xmax=1 means 1.0 -> 100%)."""
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=xmax))

def fmt_millions(x, pos):
    try:
        return f"{x/1_000_000:.0f}M"
    except Exception:
        return str(x)

def fmt_billions(x, pos):
    try:
        return f"{x/1_000_000_000:.1f}B"
    except Exception:
        return str(x)

# ---------- state shading & annotations ----------

def shade_states(
    ax,
    dates: Sequence[pd.Timestamp | str],
    states: Sequence[str],
    palette: Optional[Mapping[str, str]] = None,
    alpha: float = 0.12,
):
    """Shade contiguous runs of `states` across the x-axis."""
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    s = pd.Series(states, dtype="object").astype(str).reset_index(drop=True)
    if palette is None:
        uniq = list(pd.Series(s).dropna().unique())
        default = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
        palette = {name: default[i % len(default)] for i, name in enumerate(uniq)}
    cur = s.iloc[0]; start = d.iloc[0]
    for i in range(1, len(s)):
        if s.iloc[i] != cur:
            ax.axvspan(start, d.iloc[i], color=palette.get(cur, "#999999"), alpha=alpha, lw=0)
            cur = s.iloc[i]; start = d.iloc[i]
    ax.axvspan(start, d.iloc[-1], color=palette.get(cur, "#999999"), alpha=alpha, lw=0)
    # legend swatches
    uniq = list(pd.Series(s).dropna().unique())
    handles = [plt.Line2D([0], [0], color=palette[k], lw=6) for k in uniq]
    ax.legend(handles, uniq, title="State", fontsize=8, loc="upper left")

def annotate_min_max_latest(
    ax,
    dates: Sequence[pd.Timestamp | str],
    series: pd.Series,
    fontsize: int = 9,
):
    """Mark min, max, and last non-NaN with date and percent value if series looks like a percent."""
    s = pd.Series(series).dropna()
    if s.empty:
        return
    idx_min, idx_max, idx_last = s.idxmin(), s.idxmax(), s.index[-1]
    pts = [("Min", idx_min), ("Max", idx_max), ("Latest", idx_last)]
    date_series = pd.to_datetime(pd.Series(dates))
    sd = s.std() or 0.0
    for tag, idx in pts:
        x = date_series.iloc[idx]; y = s.loc[idx]
        ax.scatter([x], [y], s=30)
        dy = (sd * 0.15) * (+1 if tag != "Max" else -1)
        ax.annotate(
            f"{tag}\n{pd.to_datetime(x).date()}\n{y:.2%}" if s.abs().max() < 5 else f"{tag}\n{pd.to_datetime(x).date()}\n{y:.4f}",
            xy=(x, y),
            xytext=(x, y + dy),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", lw=1),
            fontsize=fontsize,
            ha="center",
            va="bottom" if dy >= 0 else "top",
        )

def add_hlines_with_labels(
    ax,
    items: Sequence[Tuple[float, str, str]],
    x_right_pos: float = 0.92,
    dates: Optional[Sequence[pd.Timestamp | str]] = None,
    legend: bool = True,
):
    """
    Draw horizontal lines and label them.
    items: list of (y_value, label, linestyle)
    If dates provided, also add inline labels near right edge.
    """
    handles = []
    for y, text, style in items:
        h = ax.axhline(y=y, linestyle=style, linewidth=1, label=text)
        handles.append(h)
    if dates is not None:
        d = pd.to_datetime(pd.Series(dates))
        x0, x1 = d.iloc[0], d.iloc[-1]
        x_ann = x0 + (x1 - x0) * x_right_pos
        for y, text, _ in items:
            ax.annotate(text, xy=(x_ann, y), xycoords="data",
                        xytext=(5, 0), textcoords="offset points",
                        fontsize=8, va="center", ha="left")
    if legend:
        ax.legend(title="Reference lines", fontsize=8)

# ---------- common plot recipes ----------

def ts_with_bands(
    dates: Sequence[pd.Timestamp | str],
    series: pd.Series,
    title: str,
    ylabel: str,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    percent_axis: bool = False,
    out: Optional[Path] = None,
):
    """Quick time series with horizontal mean / ±1σ / ±2σ bands."""
    d = pd.to_datetime(pd.Series(dates))
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean() if mean is None else mean
    sd = s.std() if std is None else std

    fig, ax = plt.subplots()
    ax.plot(d, s)
    if percent_axis:
        fmt_percent_axis(ax, xmax=1.0)
    ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel(ylabel)
    lines = [
        (0.0 if percent_axis else None, "Zero (0%)" if percent_axis else "Zero", "-"),
        (mu,  f"Mean ({mu:.2%})" if percent_axis else f"Mean ({mu:.4f})", ":" ),
        (mu+sd,  "Mean +1σ", "--"),
        (mu-sd,  "Mean -1σ", "--"),
        (mu+2*sd,"Mean +2σ", "-."),
        (mu-2*sd,"Mean -2σ", "-."),
    ]
    items = [(y, label, style) for (y, label, style) in lines if y is not None]
    add_hlines_with_labels(ax, items, dates=d)
    annotate_min_max_latest(ax, d, s)
    if out is not None:
        plot_and_save(out)
    else:
        return fig, ax

def heatmap(
    M: np.ndarray | pd.DataFrame,
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    title: str,
    out: Optional[Path] = None,
):
    """Simple heatmap utility (values in M are shown as image)."""
    Z = M.values if isinstance(M, pd.DataFrame) else np.asarray(M)
    fig, ax = plt.subplots()
    im = ax.imshow(Z, origin="lower", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(xlabels))), ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels))), ax.set_yticklabels(ylabels)
    ax.set_title(title)
    if out is not None:
        plot_and_save(out)
    else:
        return fig, ax
