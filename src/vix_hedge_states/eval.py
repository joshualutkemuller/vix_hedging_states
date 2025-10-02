# src/vix_hedge_states/eval.py
from __future__ import annotations
import pandas as pd
import numpy as np

__all__ = ["compute_metrics"]

def _max_drawdown(level: pd.Series) -> float:
    s = pd.to_numeric(level, errors="coerce")
    peak = s.cummax()
    dd = s / peak - 1.0
    return float(dd.min())

def _exp_shortfall(ret: pd.Series, alpha: float) -> float:
    r = pd.to_numeric(ret, errors="coerce").dropna()
    if len(r) == 0:
        return np.nan
    q = np.quantile(r, 1 - alpha)  # e.g., alpha=0.95 -> 5th pct
    tail = r[r <= q]
    return float(tail.mean()) if len(tail) else np.nan

def compute_metrics(nav: pd.Series, es_alpha: float = 0.95, es_alpha_tail: float = 0.99) -> dict:
    nav = pd.to_numeric(nav, errors="coerce")
    ret = nav.pct_change().dropna()

    vol_ann = float(ret.std() * np.sqrt(252))
    es95 = _exp_shortfall(ret, es_alpha)
    es99 = _exp_shortfall(ret, es_alpha_tail)
    maxdd = _max_drawdown(nav)
    # CAGR
    if len(nav) >= 2:
        years = len(nav) / 252.0
        cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0)
    else:
        cagr = np.nan

    return {"ES95": es95, "ES99": es99, "MaxDD": maxdd, "Vol": vol_ann, "CAGR": cagr}
