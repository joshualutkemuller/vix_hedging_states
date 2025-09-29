# src/vix_hedge_states/pricer.py
from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "norm_pdf", "norm_cdf",
    "bs_price_greeks",
    "bs_put", "bs_call",
    "implied_vol_bisect",
    "variance_blend_term_vol",
    "guess_term_vol_from_curve",
]

# ---------- Normal helpers ----------

def norm_pdf(x: float | np.ndarray) -> float | np.ndarray:
    x = np.asarray(x)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi) if x.ndim else float(
        math.exp(-0.5 * float(x) ** 2) / math.sqrt(2.0 * math.pi)
    )

def norm_cdf(x: float | np.ndarray) -> float | np.ndarray:
    # accurate enough for pricing; avoids external dependencies
    if np.isscalar(x):
        return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))
    x = np.asarray(x)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))

# ---------- Black–Scholes pricing ----------

def bs_price_greeks(
    S: float, K: float, T: float, sigma: float,
    r: float = 0.0, q: float = 0.0,
    kind: Literal["call", "put"] = "put",
) -> Tuple[float, float, float, float, float]:
    """
    Black–Scholes price and greeks.

    Returns
    -------
    price, delta, gamma, vega, theta  (theta is per-year)
    """
    S = float(S); K = float(K); T = max(1e-8, float(T))
    sigma = max(1e-8, float(sigma)); r = float(r); q = float(q)
    vsqrt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vsqrt
    d2 = d1 - vsqrt

    Nd1 = norm_cdf(d1); Nd2 = norm_cdf(d2)
    n_d1 = norm_pdf(d1)

    disc_r = math.exp(-r * T); disc_q = math.exp(-q * T)

    if kind == "call":
        price = S * disc_q * Nd1 - K * disc_r * Nd2
        delta = disc_q * Nd1
    else:  # put
        price = K * disc_r * norm_cdf(-d2) - S * disc_q * norm_cdf(-d1)
        delta = disc_q * (Nd1 - 1.0)

    gamma = disc_q * n_d1 / (S * vsqrt)
    vega = S * disc_q * n_d1 * math.sqrt(T)  # per 1.0 vol (i.e., 100 vol points)
    # theta (per-year; sign is price decay)
    if kind == "call":
        theta = (-S * disc_q * n_d1 * sigma / (2 * math.sqrt(T))
                 + q * S * disc_q * Nd1 - r * K * disc_r * Nd2)
    else:
        theta = (-S * disc_q * n_d1 * sigma / (2 * math.sqrt(T))
                 - q * S * disc_q * norm_cdf(-d1) + r * K * disc_r * norm_cdf(-d2))
    return price, delta, gamma, vega, theta

def bs_put(S: float, K: float, T: float, sigma: float, r: float = 0.0, q: float = 0.0):
    return bs_price_greeks(S, K, T, sigma, r, q, "put")[0]

def bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0, q: float = 0.0):
    return bs_price_greeks(S, K, T, sigma, r, q, "call")[0]

# ---------- Implied vol (bisection) ----------

def implied_vol_bisect(
    target_price: float,
    S: float, K: float, T: float,
    r: float = 0.0, q: float = 0.0,
    kind: Literal["call", "put"] = "put",
    tol: float = 1e-6, max_iter: int = 100, lo: float = 1e-4, hi: float = 5.0,
) -> float:
    """
    Solve for sigma that matches target_price using bisection.
    """
    f = (bs_call if kind == "call" else bs_put)
    plo = f(S, K, T, lo, r, q) - target_price
    phi = f(S, K, T, hi, r, q) - target_price
    if plo * phi > 0:
        # expand bracket a bit
        for h in [10.0, 20.0]:
            phi = f(S, K, T, hi * h, r, q) - target_price
            if plo * phi <= 0:
                hi *= h; break
        else:
            return np.nan
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        pm = f(S, K, T, mid, r, q) - target_price
        if abs(pm) < tol:
            return mid
        if pm * plo < 0:
            hi = mid; phi = pm
        else:
            lo = mid; plo = pm
    return 0.5 * (lo + hi)

# ---------- Term vol helpers (VIX/VX blending) ----------

def variance_blend_term_vol(
    vol1: float, T1_days: float, vol2: float, T2_days: float, target_days: float
) -> float:
    """
    Linear interpolation in *variance*time between two maturities (T1<T2).
    Returns annualized sigma for target_days.
    """
    T1 = max(1e-6, float(T1_days) / 252.0)
    T2 = max(1e-6, float(T2_days) / 252.0)
    Tt = max(1e-6, float(target_days) / 252.0)
    v1 = float(vol1) ** 2; v2 = float(vol2) ** 2
    # integrated variance at T:
    iv1 = v1 * T1
    iv2 = v2 * T2
    # interpolate IV at Tt
    if Tt <= T1:
        ivt = iv1 * (Tt / T1)  # scale down
    elif Tt >= T2:
        ivt = iv2 + (iv2 - iv1) * (Tt - T2) / (T2 - T1)  # linear extrapolation
    else:
        w = (Tt - T1) / (T2 - T1)
        ivt = iv1 * (1 - w) + iv2 * w
    return math.sqrt(max(1e-12, ivt / Tt))

def guess_term_vol_from_curve(
    df: pd.DataFrame,
    target_days: int,
    vix_col: str = "vix",
    vx1_col: str = "vx1",
    vx2_col: str = "vx2",
    days_to_vx1: int = 30,
    days_to_vx2: int = 60,
    fallback_sigma: float = 0.20,
) -> pd.Series:
    """
    Produce a daily series of term vol for a target maturity using:
      - if vx1 & vx2 exist: variance-blend between them to target_days
      - elif vix exists: use vix/100 as flat sigma
      - else: fallback constant
    """
    s_vx1 = pd.to_numeric(df[vx1_col], errors="coerce") / 100.0 if vx1_col in df.columns else None
    s_vx2 = pd.to_numeric(df[vx2_col], errors="coerce") / 100.0 if vx2_col in df.columns else None
    s_vix = pd.to_numeric(df[vix_col], errors="coerce") / 100.0 if vix_col in df.columns else None

    if s_vx1 is not None and s_vx2 is not None:
        # row-wise blend
        sig = []
        for a, b in zip(s_vx1.fillna(method="ffill"), s_vx2.fillna(method="ffill")):
            if np.isnan(a) and np.isnan(b):
                sig.append(np.nan)
            elif np.isnan(a):
                sig.append(b)
            elif np.isnan(b):
                sig.append(a)
            else:
                sig.append(variance_blend_term_vol(a, days_to_vx1, b, days_to_vx2, target_days))
        return pd.Series(sig, index=df.index).fillna(method="ffill").fillna(fallback_sigma).clip(0.05, 1.50)

    if s_vix is not None:
        return s_vix.fillna(method="ffill").fillna(fallback_sigma).clip(0.05, 1.50)

    return pd.Series(fallback_sigma, index=df.index)
