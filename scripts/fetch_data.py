# -*- coding: utf-8 -*-
"""
Placeholder module for the VIX Term-Structure States project.
Replace this with real implementation.
"""
#!/usr/bin/env python3
"""
fetch_data.py
-------------
Fetches:
- VIX (^VIX) via yfinance -> data/vix.csv
- SPY via yfinance (Adj Close) -> builds total return index -> data/spy_tr.csv
- VX1..VX4 via Nasdaq Data Link (Quandl) CHRIS continuous futures -> data/vx.csv   (WORK IN PROGRESS)
- VVIX via Nasdaq Data Link (CBOE/VVIX) if accessible -> data/vvix.csv (WORK IN PROGRESS)

Usage:
    python -m scripts.fetch_data --start 2006-01-01 --end 2025-12-31 --data_dir data

Env:
    NASDAQ_DATA_LINK_API_KEY in environment or .env (if python-dotenv is installed)
"""
import sys
from pathlib import Path

# Make src/ importable no matter where you run the script from
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "config"))  # insert(0) > append

import argparse
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from api_keys import NASDAQ_LINK_API_KEY

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def fetch_yf(symbol, start, end, include_actions=True):
    import yfinance as yf
    # auto_adjust=False ==> returns BOTH Close and Adj Close + Volume
    df = yf.download(
        symbol, start=start, end=end,
        progress=False, auto_adjust=False, actions=include_actions
    )
    df = df.reset_index().rename(columns={'Date': 'date'})
    df['date'] = pd.to_datetime(df['date']).dt.date.astype(str)
    # If actions=True, yfinance includes 'Dividends' and 'Stock Splits' columns
    if 'Stock Splits' in df.columns and 'Splits' not in df.columns:
        df = df.rename(columns={'Stock Splits': 'Splits'})
    return df

def fetch_ndl(dataset_code, start, end, api_key=None):
    import nasdaqdatalink as ndl
    if api_key:
        ndl.ApiConfig.api_key = api_key
    try:
        df = ndl.get(dataset_code, start_date=start, end_date=end)
        df = df.reset_index().rename(columns={'Date': 'date'})
        df['date'] = pd.to_datetime(df['date']).dt.date.astype(str)
        return df
    except Exception as e:
        print(f"[WARN] Failed to fetch {dataset_code}: {e}")
        return None

def build_spy_tr_from_adjclose(df_spy):
    out = df_spy[['date', 'Adj Close']].copy()
    out = out.rename(columns={'Adj Close': 'adj_close'})
    out = out.dropna().sort_values('date')
    base = out['adj_close'].iloc[0]
    out['spy_tr'] = out['adj_close'] / base * 100.0
    return out[['date', 'spy_tr']]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--start', default='1990-01-01')
    ap.add_argument('--end', default=datetime.today().strftime('%Y-%m-%d'))
    args = ap.parse_args()

    # Optional dotenv support from project root
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass

    api_key = os.getenv('NASDAQ_DATA_LINK_API_KEY')

    # Resolve data_dir relative to project root if not absolute
    raw_dir = Path(args.data_dir)
    data_dir = raw_dir if raw_dir.is_absolute() else (PROJECT_ROOT / raw_dir)
    _ensure_dir(data_dir)

    # --- VIX via yfinance (full OHLCV)
    print('[INFO] Fetching ^VIX (full OHLCV) from yfinance...')
    vix_full = fetch_yf('^VIX', args.start, args.end, include_actions=False)
    #print(vix_full)
    vix_full.to_csv(data_dir / 'vix_ohlcv.csv', index=False)
    print(f'[OK] Saved {data_dir/"vix_ohlcv.csv"} ({len(vix_full)} rows)')

    # Minimal VIX file for the project
    vix_out = vix_full[['date', 'Close']].rename(columns={'Close': 'vix'}).dropna()
    vix_out.to_csv(data_dir / 'vix.csv', index=False)
    print(f'[OK] Saved {data_dir/"vix.csv"}')

    # --- VIXX via yfinance (full OHLCV)
    print('[INFO] Fetching ^VVIX (full OHLCV) from yfinance...')
    vix_full = fetch_yf('^VVIX', args.start, args.end, include_actions=False)
    #print(vix_full)
    vix_full.to_csv(data_dir / 'vvix_ohlcv.csv', index=False)
    print(f'[OK] Saved {data_dir/"vvix_ohlcv.csv"} ({len(vix_full)} rows)')

    # Minimal VIXX file for the project
    vix_out = vix_full[['date', 'Close']].rename(columns={'Close': 'vvix'}).dropna()
    vix_out.to_csv(data_dir / 'vvix.csv', index=False)
    print(f'[OK] Saved {data_dir/"vvix.csv"}')

    # --- SPY via yfinance (full OHLCV + actions)
    print('[INFO] Fetching SPY (full OHLCV + actions) from yfinance...')
    spy_full = fetch_yf('SPY', args.start, args.end, include_actions=True)
    # Standardize column names a bit
    spy_full = spy_full.rename(columns={'Stock Splits': 'Splits'})
    spy_full.to_csv(data_dir / 'spy_ohlcv.csv', index=False)
    print(f'[OK] Saved {data_dir/"spy_ohlcv.csv"} ({len(spy_full)} rows)')

    # Build TR index for the project
    spy_tr = build_spy_tr_from_adjclose(spy_full)
    spy_tr.to_csv(data_dir / 'spy_tr.csv', index=False)
    print(f'[OK] Saved {data_dir/"spy_tr.csv"}')

    # --- VVIX (optional) via Nasdaq Data Link
    print('[INFO] Attempting VVIX from Nasdaq Data Link (CBOE/VVIX)...')
    vvix_df = fetch_ndl('CBOE/VVIX', args.start, args.end, NASDAQ_LINK_API_KEY)
    vvix_saved = False
    if vvix_df is not None:
        col = 'VVIX Close' if 'VVIX Close' in vvix_df.columns else ('Close' if 'Close' in vvix_df.columns else None)
        if col:
            vvix_out = vvix_df[['date', col]].rename(columns={col: 'vvix'}).dropna()
            vvix_out.to_csv(data_dir / 'vvix_ndl.csv', index=False)
            print(f'[OK] Saved {data_dir/"vvix_ndl.csv"} ({len(vvix_out)} rows)')
            vvix_saved = True
    if not vvix_saved:
        print('[WARN] VVIX not fetched. You can manually save data/vvix.csv with columns: date,vvix')

    # --- VX1..VX4 continuous futures via Nasdaq Data Link
    vx_cols = []
    vx_all = None
    series = [
        ('CHRIS/CBOE_VX1', 'vx1'),
        ('CHRIS/CBOE_VX2', 'vx2'),
        ('CHRIS/CBOE_VX3', 'vx3'),
        ('CHRIS/CBOE_VX4', 'vx4'),
    ]
    for code, colname in series:
        print(f'[INFO] Fetching {code}...')
        df = fetch_ndl(code, args.start, args.end, NASDAQ_LINK_API_KEY)
        if df is None:
            continue
        price_col = next((c for c in ['Settle', 'Last', 'Close'] if c in df.columns), None)
        if price_col is None:
            print(f'[WARN] {code} missing expected price columns; skipping.')
            continue
        df = df[['date', price_col]].rename(columns={price_col: colname})
        vx_cols.append(colname)
        vx_all = df if vx_all is None else vx_all.merge(df, on='date', how='outer')

    if vx_all is not None:
        vx_all = vx_all.sort_values('date')
        vx_all.to_csv(data_dir / 'vx.csv', index=False)
        print(f'[OK] Saved {data_dir/"vx.csv"} ({len(vx_all)} rows, cols: {vx_cols})')
    else:
        print('[WARN] No VX futures downloaded. Ensure NASDAQ_DATA_LINK_API_KEY is set and you have access.')

    print('[DONE] Fetch complete.')

if __name__ == '__main__':
    main()