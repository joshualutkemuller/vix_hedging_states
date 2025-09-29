# src/vix_hedge_states/hedge.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Public API used by scripts/backtest.py today
# ---------------------------------------------------------------------

@dataclass
class HedgeParams:
    """
    Parameters for the *proxy* put overlay.

    Notes
    -----
    - annual_budget: fraction of NAV you'd be willing to spend per year
      (only on days the hedge is ON). Spend is linearized per trading day.
    - sensitivity: crude knob for how much of daily *down* move the overlay
      offsets when ON (e.g., 0.7 -> offsets ~70% of down days).
    - tenor_days, moneyness_pct: kept for interface compatibility and for
      easier migration to a priced engine later; not used in the proxy math.
    """
    annual_budget: float = 0.02     # 2%/yr budget
    tenor_days: int = 126           # ~6M (not used by proxy)
    moneyness_pct: float = -0.10    # 10% OTM (not used by proxy)
    sensitivity: float = 0.7        # fraction of down move offset when ON


def simulate_put_overlay_proxy(
    dates: pd.Series,
    nav_start: float,
    ret: pd.Series,                   # simple daily return (e.g., 0.004)
    hedge_on: pd.Series,              # bool per day
    params: HedgeParams,
) -> pd.DataFrame:
    """
    Extremely fast overlay proxy used for timing studies.

    Model
    -----
    NAV_{t+1} = NAV_t * (1 + r_t) - spend_t + payoff_t
      spend_t  = (annual_budget / 252) * NAV_t           if hedge_on
               = 0                                       otherwise
      payoff_t = sensitivity * max(0, -r_t) * NAV_t      if hedge_on
               = 0                                       otherwise

    Returns
    -------
    DataFrame with columns: date, ret, hedge_on, nav, spend, payoff
    """
    # Defensive copies & alignment
    d = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "ret": pd.to_numeric(ret, errors="coerce").fillna(0.0),
        "hedge_on": hedge_on.astype(bool).fillna(False)
    }).reset_index(drop=True)

    n = len(d)
    nav = np.zeros(n, dtype=float)
    spend = np.zeros(n, dtype=float)
    payoff = np.zeros(n, dtype=float)

    nav[0] = float(nav_start)
    daily_budget = float(params.annual_budget) / 252.0
    sens = float(params.sensitivity)

    for t in range(1, n):
        nav_prev = nav[t-1]
        r = float(d.at[t, "ret"])
        on = bool(d.at[t, "hedge_on"])

        s_t = daily_budget * nav_prev if on else 0.0
        p_t = sens * max(0.0, -r) * nav_prev if on else 0.0

        nav[t] = nav_prev * (1.0 + r) - s_t + p_t
        spend[t] = s_t
        payoff[t] = p_t

    out = pd.DataFrame({
        "date": d["date"].values,
        "ret": d["ret"].values,
        "hedge_on": d["hedge_on"].values,
        "nav": nav,
        "spend": spend,
        "payoff": payoff,
    })
    return out


# ---------------------------------------------------------------------
# Optional: minimal Black–Scholes helpers (not used by default)
# ---------------------------------------------------------------------
# You can keep using the proxy above for fast iteration. If you decide to
# upgrade, these building blocks let you prototype a priced overlay without
# extra dependencies (normal CDF via math.erf).

def _norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) if np.isscalar(x) \
        else 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2.0)))

def bs_put_price_delta(
    S: float, K: float, T: float, sigma: float,
    r: float = 0.0, q: float = 0.0
) -> tuple[float, float]:
    """
    Black–Scholes European put (price, delta). No external deps.

    Parameters
    ----------
    S : spot
    K : strike
    T : years to expiry
    sigma : implied vol (absolute, e.g., 0.22)
    r : risk-free rate (cont. comp.)
    q : dividend yield (cont. comp.)

    Returns
    -------
    (price, delta)  where delta is dP/dS
    """
    S = float(S); K = float(K); T = max(1e-6, float(T)); sigma = max(1e-8, float(sigma))
    r = float(r); q = float(q)
    vol_sqrtT = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vol_sqrtT
    d2 = d1 - vol_sqrtT
    Nd1 = _norm_cdf(-d1)  # for put delta
    Nd2 = _norm_cdf(-d2)
    price = K * math.exp(-r * T) * Nd2 - S * math.exp(-q * T) * Nd1
    delta = -math.exp(-q * T) * _norm_cdf(-d1)  # put delta
    return price, delta


def simulate_put_overlay_bs(
    dates: pd.Series,
    nav_start: float,
    spot: pd.Series,                # SPY price or TR level (use price if you model q separately)
    hedge_on: pd.Series,            # bool
    tenor_days: int = 126,
    moneyness_pct: float = -0.10,
    annual_budget: float = 0.02,
    iv: Optional[pd.Series] = None, # daily IV (abs, e.g., 0.20). If None, uses VIX/100 clipped.
    r: float = 0.0,
    q: float = 0.0,
    bid_ask_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Lightweight priced overlay (optional). Opens a single put and marks to market
    daily until expiry; then rolls if hedge remains ON. Premium paid upfront.

    Sizing rule (simple):
      When we (re)open a put, we target a *one-shot* premium spend equal to:
          daily_budget * expected_holding_days * NAV
      where daily_budget = annual_budget / 252 and expected_holding_days = tenor_days.
      This roughly matches the proxy's *average* spend rate.

    IV source:
      - If `iv` is provided, we use it.
      - Else we approximate sigma_t = clip(VIX_t/100, 5%, 150%) if a `vix` column is present
        in the assembled dataframe for the same dates; otherwise we fall back to 20% flat.
    """
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "S": pd.to_numeric(spot, errors="coerce")
    }).reset_index(drop=True)
    n = len(df)

    # crude IV guess if none provided
    if iv is None:
        # if caller later merges vix, they can pass iv explicitly; for now default
        sigma = pd.Series(0.20, index=df.index, dtype=float)
    else:
        sigma = pd.to_numeric(iv, errors="coerce").fillna(method="ffill").fillna(0.20)
    sigma = sigma.clip(lower=0.05, upper=1.50)  # sanity bounds

    on = hedge_on.astype(bool).reindex(df.index, fill_value=False).values
    S = df["S"].values

    nav = np.zeros(n, dtype=float); nav[0] = float(nav_start)
    spend = np.zeros(n, dtype=float)
    payoff_or_mtm = np.zeros(n, dtype=float)

    # current option state
    has_position = False
    K = 0.0; T_days = 0; pos_price = 0.0; units = 0.0  # units = number of puts per 1 NAV (normalized)

    daily_budget = annual_budget / 252.0
    ba = float(bid_ask_bps)

    for t in range(1, n):
        nav_prev = nav[t-1]
        S_t = float(S[t-1]) if S[t-1] == S[t-1] else S[t-2] if t > 1 else S[t]  # last good
        S_tp1 = float(S[t]) if S[t] == S[t] else S_t

        # mark existing position (if any)
        mtm = 0.0
        if has_position:
            T_days = max(0, T_days - 1)
            T_yrs = max(1e-6, T_days / 252.0)
            P_new, _ = bs_put_price_delta(S_tp1, K, T_yrs, sigma[t], r, q)
            mtm = (P_new - pos_price) * units
            pos_price = P_new

        # expiry settle & clear
        if has_position and T_days == 0:
            intrinsic = max(0.0, K - S_tp1)
            mtm += (intrinsic - pos_price) * units  # realize intrinsic vs last model price
            has_position = False; K = 0.0; pos_price = 0.0; units = 0.0

        # open/roll if hedge is ON and no position
        if on[t] and (not has_position):
            # choose strike/tenor
            T_days = int(tenor_days)
            T_yrs = max(1e-6, T_days / 252.0)
            K = S_tp1 * (1.0 + float(moneyness_pct))
            P0, _ = bs_put_price_delta(S_tp1, K, T_yrs, sigma[t], r, q)

            # size so upfront spend ~= daily_budget * tenor_days * NAV
            target_prem = daily_budget * tenor_days * nav_prev
            units = 0.0 if P0 <= 0 else target_prem / P0

            # upfront spend + bid/ask
            fee = ba * target_prem
            spend[t] += (target_prem + fee)
            pos_price = P0
            has_position = True

        # update NAV with MTM (or zero), then apply underlying move
        payoff_or_mtm[t] = mtm
        # first apply underlying move (unhedged), then add mtm/spend; ordering choice is minor daily
        r_tp1 = (S_tp1 / S_t - 1.0) if (S_t and S_tp1 and S_t > 0) else 0.0
        nav[t] = nav_prev * (1.0 + r_tp1) + mtm - spend[t]

        # if hedge is OFF and we still have a position, we *hold to expiry* here
        # (simple behavior). You could also choose to liquidate immediately.

    return pd.DataFrame({
        "date": df["date"],
        "nav": nav,
        "spend": spend,
        "mtm_or_settle": payoff_or_mtm,
        "has_position": has_position
    })
