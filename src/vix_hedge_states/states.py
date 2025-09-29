# src/vix_hedge_states/states.py
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import NotFittedError
    import joblib
except Exception as e:
    raise ImportError("scikit-learn and joblib are required for states.py") from e


# ---------- Utilities ----------

def _safe_matrix(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[np.ndarray, pd.Series]:
    """Return X (no NaNs) and a mask of valid rows (index-aligned to df)."""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        raise ValueError("No feature columns found in DataFrame.")
    X = df[cols].copy()
    mask = ~X.isna().any(axis=1)
    X = X.loc[mask, :].astype(float).values
    return X, mask

def _order_by_key(df: pd.DataFrame, labels: np.ndarray, key: Optional[str]) -> List[int]:
    """
    Order cluster labels by mean of a key column (ascending).
    If key is None or missing, return sorted unique label IDs.
    """
    uniq = np.unique(labels)
    if key is None or key not in df.columns:
        return list(uniq)
    out = []
    for k in uniq:
        m = df.loc[labels == k, key].mean()
        out.append((k, m))
    out.sort(key=lambda t: (np.inf if pd.isna(t[1]) else t[1]))
    return [k for k, _ in out]

def transition_matrix(states: Sequence[str], order: Optional[Sequence[str]] = None) -> pd.DataFrame:
    s = pd.Series(states).dropna().astype(str).reset_index(drop=True)
    if s.empty:
        return pd.DataFrame()
    if order is None:
        order = sorted(s.unique().tolist())
    idx = {name: i for i, name in enumerate(order)}
    n = len(order)
    M = np.zeros((n, n), dtype=float)
    for a, b in zip(s.iloc[:-1], s.iloc[1:]):
        M[idx[a], idx[b]] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        row_sums = M.sum(axis=1, keepdims=True)
        P = np.divide(M, row_sums, where=row_sums > 0)
    return pd.DataFrame(P, index=order, columns=order)

def dwell_times(states: Sequence[str]) -> pd.DataFrame:
    s = pd.Series(states).dropna().astype(str).reset_index(drop=True)
    if s.empty:
        return pd.DataFrame(columns=["state", "length"])
    runs = []
    cur, length = s.iloc[0], 1
    for v in s.iloc[1:]:
        if v == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur, length = v, 1
    runs.append((cur, length))
    return pd.DataFrame(runs, columns=["state", "length"])


# ---------- Model classes ----------

@dataclass
class KMeansStateModel:
    k: int = 4
    scale: bool = True
    random_state: int = 42
    n_init: int = 25

    # learned artifacts
    cols_: Optional[List[str]] = None
    scaler_: Optional[StandardScaler] = None
    km_: Optional[KMeans] = None

    def fit(self, df: pd.DataFrame, cols: Sequence[str]) -> "KMeansStateModel":
        X, mask = _safe_matrix(df, cols)
        if len(X) < self.k:
            raise ValueError(f"Not enough non-NaN rows ({len(X)}) to fit KMeans(k={self.k}).")
        self.cols_ = [c for c in cols if c in df.columns]
        if self.scale:
            self.scaler_ = StandardScaler()
            X_fit = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            X_fit = X
        self.km_ = KMeans(n_clusters=self.k, n_init=self.n_init, random_state=self.random_state)
        self.km_.fit(X_fit)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.km_ is None or self.cols_ is None:
            raise NotFittedError("KMeansStateModel is not fitted.")
        X, mask = _safe_matrix(df, self.cols_)
        if self.scale and self.scaler_ is not None:
            Xp = self.scaler_.transform(X)
        else:
            Xp = X
        y = np.full(len(df), fill_value=np.nan, dtype=float)
        y[mask.values] = self.km_.predict(Xp)
        return y

    # persistence
    def save(self, path: str | Path) -> None:
        obj = {
            "k": self.k,
            "scale": self.scale,
            "random_state": self.random_state,
            "n_init": self.n_init,
            "cols_": self.cols_,
            "scaler_": self.scaler_,
            "km_": self.km_,
        }
        joblib.dump(obj, path)

    @classmethod
    def load(cls, path: str | Path) -> "KMeansStateModel":
        obj = joblib.load(path)
        m = cls(k=obj["k"], scale=obj["scale"], random_state=obj["random_state"], n_init=obj["n_init"])
        m.cols_ = obj["cols_"]
        m.scaler_ = obj["scaler_"]
        m.km_ = obj["km_"]
        return m


# Optional HMM (if hmmlearn is available)
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False

@dataclass
class HMMStateModel:
    k: int = 4
    covariance_type: str = "full"
    n_iter: int = 200
    random_state: int = 42
    scale: bool = True

    cols_: Optional[List[str]] = None
    scaler_: Optional[StandardScaler] = None
    hmm_: Optional["GaussianHMM"] = None

    def fit(self, df: pd.DataFrame, cols: Sequence[str]) -> "HMMStateModel":
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn not installed. pip install hmmlearn to use HMMStateModel.")
        X, mask = _safe_matrix(df, cols)
        self.cols_ = [c for c in cols if c in df.columns]
        if self.scale:
            self.scaler_ = StandardScaler().fit(X)
            X_fit = self.scaler_.transform(X)
        else:
            self.scaler_ = None
            X_fit = X
        self.hmm_ = GaussianHMM(n_components=self.k, covariance_type=self.covariance_type,
                                n_iter=self.n_iter, random_state=self.random_state)
        self.hmm_.fit(X_fit)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.hmm_ is None or self.cols_ is None:
            raise NotFittedError("HMMStateModel is not fitted.")
        X, mask = _safe_matrix(df, self.cols_)
        Xp = self.scaler_.transform(X) if (self.scale and self.scaler_ is not None) else X
        y = np.full(len(df), np.nan, dtype=float)
        if len(Xp) > 0:
            y_seq = self.hmm_.predict(Xp)
            y[mask.values] = y_seq
        return y


# ---------- Public helper functions (compat with your scripts) ----------

def fit_states_kmeans(df: pd.DataFrame, cols: Sequence[str], k: int = 4,
                      random_state: int = 42, scale: bool = True) -> KMeansStateModel:
    """Fit a KMeansStateModel and return it."""
    return KMeansStateModel(k=k, scale=scale, random_state=random_state).fit(df, cols)

def predict_states_kmeans(df: pd.DataFrame, cols: Sequence[str], model: Optional[KMeansStateModel] = None,
                          k: int = 4, random_state: int = 42, scale: bool = True) -> np.ndarray:
    """
    Predict states. If `model` is None, fit a fresh one (not recommended for backtests).
    Provided for convenience.
    """
    if model is None:
        model = fit_states_kmeans(df, cols, k=k, random_state=random_state, scale=scale)
    return model.predict(df)

def map_labels_to_names(labels: Iterable[int | float], names: Sequence[str]) -> np.ndarray:
    """
    Map integer cluster labels to provided human-readable names by label order.
    If there are more labels than names, names repeat modulo length.
    """
    lab = pd.Series(labels).astype("Int64")
    uniq = sorted([int(x) for x in pd.unique(lab.dropna())])
    if not names:
        names = [f"state_{i}" for i in range(len(uniq))]
    lut = {u: names[i % len(names)] for i, u in enumerate(uniq)}
    out = lab.map(lut)
    return out.astype(object).values


# ---------- Walk-forward labeling ----------

def walk_forward_kmeans(
    df: pd.DataFrame,
    cols: Sequence[str],
    k: int = 4,
    random_state: int = 42,
    scale: bool = True,
    fit_window: Optional[int] = 252 * 5,   # use last ~5y by default; None => expanding
    step: int = 21,                        # refit cadence (monthly-ish)
    min_fit_rows: int = 252 * 2,          # need at least ~2y to fit
) -> pd.Series:
    """
    Walk-forward regime labeling:
    - Fit on [t0, t_fit_end], predict on (t_fit_end, t_fit_end+step]
    - Advance by `step` and repeat.
    Returns a Series aligned to df.index with integer labels (NaN where not predicted).
    """
    cols = [c for c in cols if c in df.columns]
    if not cols:
        raise ValueError("No feature columns found for walk_forward_kmeans().")

    y = pd.Series(index=df.index, dtype="float")
    n = len(df)
    t = 0

    while t < n:
        fit_end = min(t + step*5, n-1)  # start-up heuristic: grab some initial span
        if fit_window is None:
            fit_start = 0
        else:
            fit_start = max(0, fit_end - fit_window)

        df_fit = df.iloc[fit_start:fit_end+1]
        try:
            X_fit, mask_fit = _safe_matrix(df_fit, cols)
        except ValueError:
            t += step
            continue

        if len(X_fit) < min_fit_rows:
            t += step
            continue

        model = KMeansStateModel(k=k, scale=scale, random_state=random_state).fit(df_fit, cols)

        # Predict next window (exclusive of fit_end)
        pred_start = fit_end + 1
        pred_end = min(fit_end + step, n-1)
        if pred_start > pred_end:
            break

        df_pred = df.iloc[pred_start:pred_end+1]
        y.iloc[pred_start:pred_end+1] = model.predict(df_pred)

        t = pred_end + 1

    return y


# ---------- Optional: label naming by curve severity ----------

def name_by_curve_severity(df: pd.DataFrame, labels: np.ndarray,
                           prefer_keys: Sequence[str] = ("slope12", "basis1", "vix")) -> List[str]:
    """
    Heuristic naming helper: sort clusters from 'carry/contango' to 'stress/backwardation'
    by keys (first key that exists). Returns suggested ordered names list (length = n_clusters).
    """
    key = next((k for k in prefer_keys if k in df.columns), None)
    order = _order_by_key(df, labels, key)
    # low->high order; you can customize names as you like
    template = ["carry_contango", "flat", "stress_backwardation", "convex_stress",
                "extreme_stress", "super_contango"]
    # map numeric label -> rank
    rank = {lab: i for i, lab in enumerate(order)}
    k = len(order)
    names = []
    for i in range(k):
        # find numeric label i's rank; if missing, fallback to index
        r = rank.get(i, i)
        names.append(template[r % len(template)])
    return names
