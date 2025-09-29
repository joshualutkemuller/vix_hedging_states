import sys
from pathlib import Path

# Make src/ importable no matter where you run the script from
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
sys.path.insert(0, str(PROJECT_ROOT / "config"))  # insert(0) > append

#!/usr/bin/env python3
import os, sys
from pathlib import Path
import pandas as pd
import time
import argparse
import nasdaqdatalink as ndl
from api_keys import NASDAQ_LINK_API_KEY
from nasdaqdatalink.model.dataset import Dataset
from nasdaqdatalink.model.merged_dataset import MergedDataset
from nasdaqdatalink.get import get
from nasdaqdatalink.api_config import ApiConfig
from nasdaqdatalink.connection import Connection
import os
print(os.path.join(CONFIG_DIR,".corporatenasdaqdatalinkapikey"))

ndl.read_key(filename=os.path.join(CONFIG_DIR,".corporatenasdaqdatalinkapikey"))
start = "2025-01-01"
end   = "2026-12-31"

# --- load API key ---
# If your key is in ~/.nasdaq/data_link_apikey, you can skip this line.
# Otherwise point to your custom file:
# --- fetch a single VX contract: Mar-2026 (H = March) ---
code = "CHRIS/CBOE_VXF6"
#df = ndl.get(code, start_date=start, end_date=end).reset_index()
df = ndl.get('CHRIS-wiki-continuous-futures/')
print(df)
#data = ndl.get_table('ZACKS/FC', ticker='AAPL')
print(data)
hi = bi
print(df)
def _import_ndl():
    try:
        import nasdaqdatalink as ndl
        return ndl
    except Exception as e:
        print("ERROR: Please install nasdaqdatalink: pip install nasdaqdatalink", file=sys.stderr)
        raise

def _get_api_key():
    key = NASDAQ_LINK_API_KEY or os.getenv("NDL_API_KEY") or os.getenv("QUANDL_API_KEY")
    if not key:
        print("ERROR: Set NDL_API_KEY (or QUANDL_API_KEY) environment variable with your Nasdaq Data Link API key.", file=sys.stderr)
        sys.exit(2)
    return key

def fetch_one(ndl, dataset: str, start: str | None, end: str | None, retries: int = 3, sleep: float = 1.5) -> pd.DataFrame:
    last_err = None
    for _ in range(retries):
        try:
            df = ndl.get(dataset, start_date=start, end_date=end)
            # CHRIS returns Date-indexed frame with columns like: Open, High, Low, Settle, Change, Volume, Previous Day Open Interest
            df = df.reset_index()
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            if "settle" not in df.columns:
                raise ValueError(f"{dataset} missing 'Settle' column. Got: {df.columns.tolist()}")
            df = df.rename(columns={"date": "date", "settle": "settle"})
            df["date"] = pd.to_datetime(df["date"])
            return df[["date", "settle"]]
        except Exception as e:
            last_err = e
            time.sleep(sleep)
    raise RuntimeError(f"Failed fetching {dataset}: {last_err}")

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="YYYY-MM-DD")
    ap.add_argument("--end",   default=None, help="YYYY-MM-DD")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4], help="Which VX depths to pull (e.g., 1 2 3 4)")
    ap.add_argument("--out", default=str(DATA_DIR / "vx.csv"))
    args = ap.parse_args(argv)

    ndl = _import_ndl()
    api_key = _get_api_key()
    ndl.ApiConfig.api_key = api_key

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged = None
    for n in args.depths:
        code = f"CHRIS/CBOE_VX{n}"
        print(f"[NDL] Downloading {code} ...")
        try:
            df = fetch_one(ndl, code, args.start, args.end)
        except Exception as e:
            # Common cause: dataset not available for your key (404 / NotFound)
            print(f"[WARN] Could not fetch {code}: {e}", file=sys.stderr)
            continue
        df = df.rename(columns={"settle": f"vx{n}"})
        merged = df if merged is None else merged.merge(df, on="date", how="outer")

    if merged is None or merged.empty:
        print("[ERROR] No VX datasets were fetched. "
              "This may mean your NDL key doesn’t have access to CHRIS/CBOE_VX*, "
              "or the symbols are unavailable. Consider switching to CFE daily settlements.", file=sys.stderr)
        return 1

    merged = merged.sort_values("date").reset_index(drop=True)
    merged.to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path} with columns: {merged.columns.tolist()}")
    return 0

if __name__ == "__main__":
    main()
    raise SystemExit(main())
