# directory_creator.py
# Creates the project tree and writes a bootstrap script that can (re)create blank Python files.

import os
import stat
from pathlib import Path

# ------------ always work relative to this script ------------
os.chdir(Path(__file__).resolve().parent)

root = Path(__file__).resolve().parent
pkg = root / "src" / "vix_hedge_states"
utils_dir = pkg / "utils"
config_dir = root / "config"
scripts_dir = root / "scripts"
data_dir = root / "data"
tests_dir = root / "tests"
notebooks_dir = root / "notebooks"
experiments_dir = root / "experiments"
reports_dir = root / "reports"

# Create directories
for d in [
    pkg, utils_dir, config_dir, scripts_dir, data_dir, tests_dir,
    notebooks_dir, experiments_dir, reports_dir, reports_dir / "figures", reports_dir / "tables"
]:
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# How to run bootstrap_placeholders.py 
# --------------------------------------------------------------------
"""
python -m scripts.bootstrap_placeholders           # create missing files
python -m scripts.bootstrap_placeholders --force   # overwrite with placeholders
python -m scripts.bootstrap_placeholders --dry-run # preview

"""
