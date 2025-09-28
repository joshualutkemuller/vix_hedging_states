# src/vix_hedge_states/features.py
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Iterable, Optional, Sequence, Tuple

__all__ = [
    "build_features",
    "zscore_frame",
    "realized_vol",
    "add_forward_returns",
]

# ---------------- helpers ---------------- #

def _num(s: pd.Series) -> pd.Series:
    """Coerce to numeric, keeping NaN on failures."""
    return pd.to_numeric(s, errors="coerce")

def _has(df: pd.DataFrame, cols: Sequence[str]) -> bool:
    return all(c in df.columns for c in cols)

def realized_vol(ret: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """
    Rolling realized volatility from a return series (log or simple).
    Returns percent units (e.g., 18.5 == 18.5%).
    """
    vol = ret.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol * 100.0

def add_forward_returns(
    df: pd.DataFrame,
    price_col: str,
    horizons: Sequence[int] = (5, 20),
    prefix: str = "fwd_ret_",
    log: bool = False,
) -> pd.DataFrame:
    """Append forward returns for quick EDA/regressions."""
    if price_col not in df.columns:
        return df
    if log:
        x = np.log(_num(df[price_col]))
        for h in horizons:
            df[f"{prefix}{h}d"] = (x.shift(-h) - x)
    else:
        for h in horizons:
            df[f"{prefix}{h}d"] = _num(df[price_col]).pct_change(h).shift(-h)
    return df

# ---------------- main feature builder ---------------- #

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build core term-structure & vol features.

    Inputs (if present):
      - 'vix' (spot)
      - 'vx1','vx2','vx3','vx4' (VIX futures levels)
      - 'vvix' (optional)
      - 'spy_tr' (SPY total return index level; any base)

    Outputs (added columns if inputs present):
      - basis1    = vx1 - vix
      - slope12   = vx2 - vx1
      - slope13   = vx3 - vx1
      - curvature = vx2 - 0.5*(vx1 + vx3)
      - dvix_5, dvix_20 (5/20-day changes)
      - ret       (log return of spy_tr)
      - rv20      (20-day realized vol from ret, annualized %, in points)
    """
    d = df.copy()

    # Term-structure features (defensive to missing cols)
    if _has(d, ["vx1", "vix"]):
        d["basis1"] = _num(d["vx1"]) - _num(d["vix"])
    if _has(d, ["vx2", "vx1"]):
        d["slope12"] = _num(d["vx2"]) - _num(d["vx1"])
    if _has(d, ["vx3", "vx1"]):
        d["slope13"] = _num(d["vx3"]) - _num(d["vx1"])
    if _has(d, ["vx2", "vx1", "vx3"]):
        d["curvature"] = _num(d["vx2"]) - 0.5 * (_num(d["vx1"]) + _num(d["vx3"]))

    # VIX momentum
    if "vix" in d.columns:
        vixn = _num(d["vix"])
        d["dvix_5"] = vixn.diff(5)
        d["dvix_20"] = vixn.diff(20)

    # SPY log returns & realized vol proxy
    if "spy_tr" in d.columns:
        st = _num(d["spy_tr"])
        d["ret"] = np.log(st).diff()
        d["rv20"] = realized_vol(d["ret"], window=20, annualize=True)

    return d

# ---------------- z-score utility ---------------- #

def zscore_frame(
    df: pd.DataFrame,
    cols: Iterable[str],
    win: int = 1260,
    min_periods: int = 252,
    suffix: str = "_z",
    clip: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """
    Add rolling z-scores for selected columns.

    Parameters
    ----------
    df : DataFrame
    cols : iterable of column names to z-score
    win : rolling window length (e.g., ~5y = 1260)
    min_periods : minimum lookback before computing z
    suffix : appended to new columns
    clip : optional (lo, hi) clipping of z-scores
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = _num(out[c])
        mu = s.rolling(win, min_periods=min_periods).mean()
        sd = s.rolling(win, min_periods=min_periods).std(ddof=0)
        z = (s - mu) / sd
        if clip is not None:
            z = z.clip(lower=clip[0], upper=clip[1])
        out[f"{c}{suffix}"] = z
    return out
