# src/vix_hedge_states/io.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict

__all__ = ["load_all_data"]

# ---------------- helpers ---------------- #

def _ensure_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Ensure a datetime index column named `date` and sort ascending."""
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    return df.sort_values(col).reset_index(drop=True)

def _maybe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    """Return DataFrame if CSV loads, else None (no exception)."""
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def _gen_synthetic(n: int = 5000, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a synthetic dataset so the pipeline can run without real files."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2005-01-03", periods=n)

    # Synthetic SPY TR (geometric random walk)
    mu = 0.07 / 252
    sigma = 0.18 / np.sqrt(252)
    shocks = rng.normal(mu, sigma, size=n)
    tr = 100 * np.exp(np.cumsum(shocks))
    spy = pd.DataFrame({"date": dates, "spy_tr": tr})

    # Synthetic VIX & VX futures (with occasional spikes)
    vix = np.clip(15 + rng.normal(0, 1.5, n).cumsum() / 20 + rng.normal(0, 2.0, n), 10, 80)
    spikes = rng.random(n) < 0.01
    vix[spikes] += rng.uniform(10, 30, spikes.sum())
    vx1 = vix + rng.normal(1.0, 1.5, n)
    vx2 = vix + rng.normal(2.0, 1.7, n) + 0.5
    vx3 = vix + rng.normal(3.0, 2.0, n) + 1.0
    vx4 = vix + rng.normal(4.0, 2.0, n) + 1.5
    vx = pd.DataFrame({"date": dates, "vx1": vx1, "vx2": vx2, "vx3": vx3, "vx4": vx4})
    vix_df = pd.DataFrame({"date": dates, "vix": vix})
    vvix = pd.DataFrame({"date": dates, "vvix": vix * 1.2 + rng.normal(0, 3, n)})

    return vix_df, vvix, vx, spy

# ---------------- main loader ---------------- #

def load_all_data(data_dir: str | Path, cfg: dict) -> pd.DataFrame:
    """
    Load datasets per config; merge ONLY those that exist.
    Honors `data.required` + `data.strict` and sanitizes numeric columns.

    config example:
      data:
        vix_file: vix.csv
        vvix_file: vvix.csv
        vx_file: vx.csv
        spy_tr_file: spy_tr.csv
        strict: true                         # if True, missing required -> raise
        required: ["vix","vx","spy_tr"]      # logical names to require (vvix optional)

    Returns
    -------
    DataFrame with a `date` column and whichever of {vix, vvix, vx1..vx4, spy_tr} are available.
    """
    data_dir = Path(data_dir)
    dcfg: Dict = cfg.get("data", {})
    strict: bool = bool(dcfg.get("strict", True))
    required = set(dcfg.get("required", ["vix", "vx", "spy_tr"]))

    # file map
    files = {
        "vix":    data_dir / dcfg.get("vix_file", "vix.csv"),
        "vvix":   data_dir / dcfg.get("vvix_file", "vvix.csv"),
        "vx":     data_dir / dcfg.get("vx_file",  "vx.csv"),
        "spy_tr": data_dir / dcfg.get("spy_tr_file", "spy_tr.csv"),
    }

    # read what exists
    dfs = {
        "vix": _maybe_read_csv(files["vix"]),
        "vvix": _maybe_read_csv(files["vvix"]),
        "vx": _maybe_read_csv(files["vx"]),
        "spy_tr": _maybe_read_csv(files["spy_tr"]),
    }

    # enforce required set
    missing_required = [name for name in required if dfs.get(name) is None]
    if missing_required:
        if strict:
            pretty = ", ".join(f"{m} ({files[m].name})" for m in missing_required)
            raise FileNotFoundError(
                f"Missing required data: {pretty}. "
                "Provide these files, adjust data.required, or set data.strict: false."
            )
        # non-strict: synthetic fallback for all
        vix_df, vvix_df, vx_df, spy_df = _gen_synthetic()
        dfs.update({"vix": vix_df, "vvix": vvix_df, "vx": vx_df, "spy_tr": spy_df})

    # normalize ONLY those that exist
    for k in ("vix", "vvix", "vx", "spy_tr"):
        if dfs[k] is not None:
            dfs[k] = _ensure_datetime(dfs[k])

    # merge only present frames; prefer starting base: vix -> spy_tr -> vx -> vvix
    merge_order = [name for name in ("vix", "spy_tr", "vx", "vvix") if dfs.get(name) is not None]
    if not merge_order:
        raise ValueError("No datasets available to merge.")

    df = dfs[merge_order[0]].copy()
    for name in merge_order[1:]:
        df = df.merge(dfs[name], on="date", how="inner")

    # sanitize numerics (handles stray strings like '^VIX')
    num_cols = ["vix", "vvix", "vx1", "vx2", "vx3", "vx4", "spy_tr"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows where all key drivers are NaN; keep timeline tidy
    key_any = [c for c in ["vix", "spy_tr", "vx1", "vx2", "vx3"] if c in df.columns]
    if key_any:
        df = df.dropna(subset=key_any, how="all")

    # fill vvix if present
    if "vvix" in df.columns:
        df["vvix"] = df["vvix"].ffill().bfill()

    df = df.sort_values("date").reset_index(drop=True)
    return df
