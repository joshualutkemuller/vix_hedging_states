# VIX Term‑Structure States for Hedging Timing

Config‑driven Python project that studies whether the **shape and dynamics of the VIX futures curve**
can be used to **time index put overlays** more efficiently than a **static premium budget**.

The code builds VIX term‑structure features, discovers **volatility states** (K‑means baseline),
and turns a **SPY put hedge** on/off based on state‑conditioned rules. It backtests cost, drawdown
reduction, and **Expected Shortfall (ES95/ES99)**, then reports **hedge efficiency**
(loss reduced per unit premium).

---

## ✨ Key Features
- **Synthetic or real data**: runs immediately with synthetic series; drops in with your CSV histories.
- **Config‑first**: all parameters in `config/config.yaml` (states, policy, costs, dates).
- **Unsupervised states**: K‑means on term‑structure features; HMM stub ready for future work.
- **Option pricing**: simple Black–Scholes with IV proxied from VIX and tenor.
- **Backtest engine**: monthly rebalance cadence, premium ledger, simplified MTM.
- **Reporting**: CSV of timeseries and a NAV plot under `reports/`.

> This is a compact research scaffold meant for coursework / prototyping. For production,
replace the option surface with broker or vendor IVs, add fills/slippage models, and harden
walk‑forward state refits and roll logic.

---

## 📦 Repository Layout
```
vix_hedge_states/
  config/
    config.yaml                # main experiment configuration
  experiments/
    exp_policy_sweep.yaml      # example grid (not wired to CLI yet)
  src/vix_hedge_states/
    __init__.py
    io.py                      # data load + synthetic generator
    features.py                # term-structure features, z-scores
    states.py                  # KMeans state fit/predict + naming
    pricer.py                  # Black–Scholes, IV proxy
    hedge.py                   # rule-based on/off & sizing
    backtest.py                # engine + simple accounting
    eval.py                    # ES/MaxDD/Vol metrics
    plots.py                   # NAV plot
    utils/
      __init__.py
      dates.py, logging.py, math.py
  scripts/
    run_backtest.py            # main CLI entry point
  data/                        # drop your CSVs here (optional)
  reports/                     # outputs (figures, tables, timeseries)
  tests/
    test_smoke.py
  README.md
  requirements.txt
  pyproject.toml
```

---

## 🛠️ Installation
```bash
# (optional) in a fresh virtual environment
pip install -r requirements.txt
```

Python ≥ 3.9 is recommended.

---

## 🚀 Quick Start (synthetic data)
```bash
python -m scripts.run_backtest --config config/config.yaml
```
What happens:
1. If no CSVs under `data/`, a synthetic SPY/VIX/VX1‑4/VVIX dataset is generated.
2. Features are engineered and standardized.
3. K‑means discovers `k` volatility states; states are mapped to readable names.
4. A rule‑based hedge policy (e.g., hedge only in stress/backwardation states) trades monthly.
5. Results: metrics printed to console, NAV plot saved to `reports/figures/nav.png`,
   timeseries to `reports/backtest_timeseries.csv`.

---

## 📊 Using Real Data
Place CSVs into `data/` with these minimal schemas (ISO dates: `YYYY-MM-DD`):

- `data/vix.csv`
  ```csv
  date,vix
  2007-01-03,12.55
  ...        , ...
  ```
- `data/vvix.csv` *(optional)*
  ```csv
  date,vvix
  2012-01-03,90.1
  ...       , ...
  ```
- `data/vx.csv` *(annualized levels, percent points)*
  ```csv
  date,vx1,vx2,vx3,vx4
  2007-01-03,14.8,15.2,15.7,16.0
  ...        , ... , ... , ... , ...
  ```
- `data/spy_tr.csv` *(total return index, any base)*
  ```csv
  date,spy_tr
  2007-01-03,100.00
  ...       ,  ...
  ```

> The engine normalizes `spy_tr` to 100 at backtest start; any base is fine.

---

## ⚙️ Configuration Reference (`config/config.yaml`)
Key blocks (abridged):
```yaml
seed: 42

paths:
  data_dir: data
  reports_dir: reports
  figures_dir: reports/figures
  tables_dir: reports/tables

data:
  vix_file: vix.csv
  vvix_file: vvix.csv
  vx_file: vx.csv
  spy_tr_file: spy_tr.csv

features:
  use_columns: [vix, vvix, vx1, vx2, vx3, basis1, slope12, slope13, curvature, dvix_5, dvix_20]
  rolling_zscore_window: 1260

states:
  method: kmeans        # (hmm stubbed for later)
  k: 4
  labels: ["carry_contango", "flat", "stress_backwardation", "convex_stress"]

policy:
  annual_budget: 0.02   # 2%/yr of NAV
  hedge_states: ["stress_backwardation", "convex_stress"]
  tenor_days: 126       # ~ 6M
  moneyness_pct: -0.10  # 10% OTM put
  trigger:
    type: simple
    rules:
      require_negative_slope12: true
      require_vix_z_gt: 0.5
      require_vvix_z_gt: 0.0

costs:
  option_bid_ask_bps: 0.0015
  spy_commission_bps: 0.0001

backtest:
  start: 2007-01-01
  end: 2024-12-31
  initial_nav: 100.0
  rebalance_cadence_days: 21

evaluation:
  es_alpha: 0.95
  es_alpha_tail: 0.99
```
> Tweak these to run alternate strategies (e.g., tenor=63, moneyness=-0.05, budget=0.03).

---

## 🧠 Methodology (in brief)
1. **Features:** basis (vx1−VIX), slopes (vx2−vx1, vx3−vx1), curvature, VIX momentum, RV proxy.
2. **States:** K‑means clusters z‑scored features; clusters are mapped to intuitive regime names.
3. **Policy:** hedge only in selected states and when simple gates pass (negative slope, elevated VIX/VVIX z).
4. **Pricing:** Black–Scholes put with IV ≈ `VIX * sqrt(30/tenor_days)` (rough proxy).
5. **Backtest:** monthly portfolio actions, budget‑aware sizing, simplified MTM on expiry (illustrative).

**Metrics:** ES95/ES99, Max Drawdown, volatility, plus premium spend in the ledger.

---

## 🔁 Reproducibility
- All experiments are driven by YAML config.
- A fixed `seed` controls clustering reproducibility.
- Outputs are written to `reports/` and can be version‑controlled.

---

## 🧩 Extending the Project
- **HMM states:** add `hmmlearn` and implement sticky Gaussian HMM in `states.py`.
- **Option surface:** plug vendor IV surfaces (moneyness×tenor skew) into `pricer.py`.
- **Execution realism:** daily re‑pricing, b/a slippage scaling in stress, delta limits.
- **Policy grid:** wire `experiments/exp_policy_sweep.yaml` to a small grid‑runner to export
  heatmaps of `{ES reduction vs premium}` across tenor/moneyness/budget.

---

## 🧪 Testing
```
pytest -q
```
`tests/test_smoke.py` ensures imports succeed. Add your own tests under `tests/`.

---

## ❗ Limitations & Assumptions
- IV proxy from VIX is approximate; real surfaces recommended.
- Monthly rebalance & expiry MTM are simplified; see comments in `backtest.py`.
- State mapping is heuristic for readability; feel free to hard‑code label ordering.

---

## 📜 License
Add your preferred license (MIT/Apache‑2.0).

---

## 🙋 FAQ
**Q:** Can I run without VVIX or SKEW?  
**A:** Yes. Missing fields are forward‑filled/bypassed; remove them from `features.use_columns` or let synthetic data supply them.

**Q:** Does the backtest include dynamic delta or rolling rules?  
**A:** The scaffold re‑opens monthly; extending to delta/risk‑based triggers is encouraged.

**Q:** How do I compare against a static 2% policy?  
**A:** Set `policy.hedge_states` to `["carry_contango","flat","stress_backwardation","convex_stress"]`
to always‑on, or create a second config with `annual_budget: 0.02` and no triggers.
