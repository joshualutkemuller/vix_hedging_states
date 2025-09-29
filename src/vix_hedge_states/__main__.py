#!/usr/bin/env python3
"""
vix_hedge_states CLI

Examples:
  python -m vix_hedge_states fetch-data --start 1990-01-01 --end 2025-12-31
  python -m vix_hedge_states eda --config config/config.yaml
  python -m vix_hedge_states states-eda --config config/config.yaml
  python -m vix_hedge_states backtest --config config/config.yaml
  python -m vix_hedge_states summary --config config/config.yaml
  python -m vix_hedge_states grid --config config/config.yaml --k 3 4 5
"""
import sys
import argparse

def _call(module_name: str, argv_tail: list[str]) -> int:
    # import the script module and invoke its main(argv) if available
    mod = __import__(module_name, fromlist=["main"])
    if hasattr(mod, "main"):
        return int(mod.main(argv_tail) or 0)
    # fallback: execute as a script via runpy (rarely needed)
    import runpy
    runpy.run_module(module_name, run_name="__main__")
    return 0

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(prog="vix_hedge_states", add_help=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fetch-data", help="Download VIX/SPY/VX data into data/")
    sub.add_parser("eda", help="Run basic EDA and save tables/figures")
    sub.add_parser("states-eda", help="Fit regimes and export state reports")
    sub.add_parser("backtest", help="Run hedge backtest from config")
    sub.add_parser("summary", help="Write summary metrics & crash-capture")
    sub.add_parser("grid", help="Parameter sweep / grid search")

    # split at first subcommand boundary so we can pass through args
    if not argv:
        parser.print_help()
        return 0
    cmd = argv[0]
    tail = argv[1:]

    if cmd == "fetch-data":
        return _call("scripts.fetch_data", tail)
    if cmd == "eda":
        return _call("scripts.run_eda", tail)
    if cmd == "states-eda":
        return _call("scripts.run_states_eda", tail)
    if cmd == "backtest":
        return _call("scripts.run_backtest", tail)
    if cmd == "summary":
        return _call("scripts.report_summary", tail)
    if cmd == "grid":
        return _call("scripts.run_grid", tail)

    parser.print_help()
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
