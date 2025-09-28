# src/vix_hedge_states/backtest.py
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Sequence

from .hedge import HedgeParams, simulate_put_overlay_proxy

@dataclass
class PolicyConfig:
    hedge_states: Sequence[str]
    annual_budget: float
    tenor_days: int
    moneyness_pct: float
    trigger_rules: Dict

def _apply_triggers(df: pd.DataFrame, rules: Dict) -> pd.Series:
    ok = pd.Series(True, index=df.index)
    if not rules:
        return ok

    if rules.get("require_negative_slope12", False) and "slope12" in df:
        ok &= (pd.to_numeric(df["slope12"], errors="coerce") < 0)

    vix_thr = rules.get("require_vix_z_gt", None)
    if vix_thr is not None and "vix_z" in df:
        ok &= (pd.to_numeric(df["vix_z"], errors="coerce") > float(vix_thr))

    vvix_thr = rules.get("require_vvix_z_gt", None)
    if vvix_thr is not None and "vvix_z" in df:
        ok &= (pd.to_numeric(df["vvix_z"], errors="coerce") > float(vvix_thr))

    return ok

class Backtester:
    def __init__(self, cfg: dict):
        pol = cfg.get("policy", {})
        self.policy = PolicyConfig(
            hedge_states=pol.get("hedge_states", []),
            annual_budget=float(pol.get("annual_budget", 0.02)),
            tenor_days=int(pol.get("tenor_days", 126)),
            moneyness_pct=float(pol.get("moneyness_pct", -0.10)),
            trigger_rules=(pol.get("trigger", {}).get("rules", {}) if pol.get("trigger") else {}),
        )
        self.initial_nav = float(cfg.get("backtest", {}).get("initial_nav", 100.0))
        self.rebalance_cadence_days = int(cfg.get("backtest", {}).get("rebalance_cadence_days", 21))
        self.seed = int(cfg.get("seed", 42))
        # proxy sensitivity (optional): you can expose as cfg['policy']['sensitivity']
        self.sensitivity = float(pol.get("sensitivity", 0.7))

    def _derive_hedge_on(self, df: pd.DataFrame, states: pd.Series) -> pd.Series:
        in_state = states.astype(str).isin(set(self.policy.hedge_states))
        trig_ok = _apply_triggers(df, self.policy.trigger_rules)
        return (in_state & trig_ok)

    def run(self, df: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
        """
        df expects at least:
            'date' (datetime), 'spy_tr' (level) or 'ret' (daily returns)
            features used by triggers if configured: e.g., 'slope12', 'vix_z', 'vvix_z'
        states: array-like of state names aligned to df
        """
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        # returns: prefer simple daily return from 'spy_tr' if not provided
        if "ret" not in d.columns and "spy_tr" in d.columns:
            level = pd.to_numeric(d["spy_tr"], errors="coerce")
            d["ret"] = level.pct_change().fillna(0.0)
        elif "ret" in d.columns:
            # if it's log returns, roughly convert to simple; if already simple, this is a no-op-ish
            r = pd.to_numeric(d["ret"], errors="coerce")
            d["ret"] = (np.exp(r) - 1.0).where(r.between(-0.3, 0.3), r)  # guard crazy
        else:
            raise ValueError("Need either 'spy_tr' or 'ret' in dataframe for backtest.")

        hedge_on = self._derive_hedge_on(d, pd.Series(states, index=d.index))
        params = HedgeParams(
            annual_budget=self.policy.annual_budget,
            tenor_days=self.policy.tenor_days,
            moneyness_pct=self.policy.moneyness_pct,
            sensitivity=self.sensitivity,
        )

        bt = simulate_put_overlay_proxy(
            dates=d["date"], nav_start=self.initial_nav, ret=d["ret"], hedge_on=hedge_on, params=params
        )
        bt["state"] = states
        return bt
