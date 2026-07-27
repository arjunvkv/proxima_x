"""Run V2z_CPPF on MT5 via CLI and compare with Python backtest.

Usage:
    python research/cppf/run_mt5_bt.py                   # Run all pairs
    python research/cppf/run_mt5_bt.py --pairs EURUSD    # Single pair
    python research/cppf/run_mt5_bt.py --z 2.5           # Custom z threshold
    python research/cppf/run_mt5_bt.py --analyze         # Just re-analyze existing reports
"""
import argparse
import configparser
import subprocess
import sys
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

# --- Paths ---
MT5_TERMINAL = r"C:\Program Files\MetaTrader 5\terminal64.exe"
MQL_DIR = Path(os.environ["APPDATA"]) / "MetaQuotes" / "Terminal" / "D0E8209F77C8CF37AD8BF550E51FF075" / "MQL5"
PROFILES_DIR = MQL_DIR / "Profiles" / "Tester"
REPORTS_DIR = Path(__file__).parent / "bt_reports"
CONFIGS_DIR = Path(__file__).parent / "bt_configs"

# --- Pairs to test (matching current v2z_bar) ---
PAIRS = ["GBPNZD", "EURNZD", "GBPAUD", "EURAUD", "GBPCAD", "AUDNZD"]

def make_config(pair, z_thresh=2.5, stop_a=3.0, trig_a=0.5, gap_a=0.1,
                start="2025.01.01", end="2025.06.01") -> Path:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Write .set file
    set_name = f"v2z_bt.set"
    set_path = PROFILES_DIR / set_name
    set_content = f"""Z_THRESHOLD={z_thresh}
STOP_A={stop_a}
TRIG_A={trig_a}
GAP_A={gap_a}
MAX_HOLD_BARS=54
ATR_PERIOD=20
Z_WINDOW=50
BASE_LOT=1.0
MAX_DAILY_LOSS=1250.0
MAX_SPREAD_PIPS=50.0
MAGIC_NUMBER=202411
MAX_TRADES_DAY=500
"""
    with open(set_path, "w") as f:
        f.write(set_content)

    # Write .ini config
    report_path = REPORTS_DIR / f"{pair}_z{z_thresh:.1f}.xml"
    ini_name = f"{pair}_z{z_thresh:.1f}.ini"
    ini_path = CONFIGS_DIR / ini_name

    cfg = configparser.ConfigParser()
    cfg["Tester"] = {
        "Expert": "V2z_CPPF",
        "ExpertParameters": set_name,
        "Symbol": pair,
        "Period": "M1",
        "Model": "0",
        "FromDate": start,
        "ToDate": end,
        "Deposit": "25000",
        "Leverage": "100",
        "Optimization": "0",
        "ShutdownTerminal": "1",
        "Report": str(report_path),
    }
    with open(ini_path, "w") as f:
        cfg.write(f)

    return ini_path, report_path


def run_backtest(ini_path: Path, timeout=120) -> bool:
    """Run MT5 backtest via CLI. Returns True if completed."""
    print(f"  Running: {ini_path.stem}")
    try:
        result = subprocess.run(
            [MT5_TERMINAL, f"/config:{ini_path}"],
            capture_output=True, text=True, timeout=timeout,
        )
        return True
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after {timeout}s")
        return False
    except FileNotFoundError:
        print(f"    MT5 terminal not found at {MT5_TERMINAL}")
        return False


def parse_report(xml_path: Path) -> dict:
    """Parse MT5 XML report into summary metrics."""
    if not xml_path.exists():
        return {}

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"    Parse error: {e}")
        return {}

    ns = {"ns": "http://www.metatrader5.com/report"}

    stats = root.find(".//ns:statistics", ns)
    if stats is None:
        stats = root.find(".//statistics")

    result = {}
    if stats is not None:
        for child in stats:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            result[tag] = child.text

    # Extract trades
    trades = []
    deals = root.find(".//ns:deals", ns)
    if deals is None:
        deals = root.find(".//deals")
    if deals is not None:
        for deal in deals.findall("deal"):
            d = {}
            for attr in deal.attrib:
                d[attr] = deal.attrib[attr]
            for child in deal:
                tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                d[tag] = child.text
            trades.append(d)

    result["_trades"] = trades
    result["_n_trades"] = len(trades)
    return result


def compare_with_python(mt5_report: dict, python_metrics: dict) -> dict:
    """Compare MT5 vs Python backtest metrics."""
    comparison = {}
    for metric in ["total_net_profit", "profit_factor", "total_trades",
                   "percent_profitable", "max_drawdown_percent"]:
        mt5_val = mt5_report.get(metric, "0")
        py_val = python_metrics.get(metric, "0")
        try:
            diff = float(mt5_val) - float(py_val)
        except (ValueError, TypeError):
            diff = 0
        comparison[metric] = {"mt5": mt5_val, "python": py_val, "diff": diff}
    return comparison


def run_python_bt(pair, z_thresh=2.5) -> dict:
    """Run Python backtest and return metrics."""
    import numpy as np
    import pandas as pd
    from pathlib import Path

    PARQUET_DIR = Path(__file__).parent.parent.parent / "research" / "phase_dislocation" / "dukascopy_data"
    pair_lower = pair.lower()

    PARQUET_DIR = PARQUET_DIR.resolve()
    df = pd.read_parquet(PARQUET_DIR / f"{pair_lower}.parquet").set_index("timestamp").astype(float)
    df *= 10000  # to pips

    # Run hfdf_m1 logic
    ret = df["close"].diff()
    z = (ret - ret.shift(1).rolling(50).mean()) / ret.shift(1).rolling(50).std().clip(1e-8)
    atr = (df["high"] - df["low"]).shift(1).rolling(20).mean().clip(1e-8)
    valid = z.notna() & atr.notna()
    valid &= z.abs() >= z_thresh
    idxs = np.where(valid)[0]
    if len(idxs) < 5:
        return {"total_trades": 0, "total_net_profit": 0}

    max_bars = 54
    pnls = []
    c, h, l = df["close"].values, df["high"].values, df["low"].values
    for pos in idxs:
        if pos + 2 >= len(df): continue
        direction = -1 if z.iloc[pos] > 0 else 1
        entry = c[pos]; atr_v = atr.iloc[pos]
        s = 3.0 * atr_v; tg = 0.5 * atr_v; gp = 0.1 * atr_v
        best = entry; exited = False
        for j in range(1, max_bars + 1):
            bp = pos + j
            if bp >= len(df): break
            if direction == 1:
                if h[bp] > best: best = h[bp]
                sl = entry - s
                if best - entry > tg: sl = best - gp
                if l[bp] <= sl:
                    pnls.append((sl - entry) * direction)
                    exited = True; break
            else:
                if l[bp] < best: best = l[bp]
                sl = entry + s
                if entry - best > tg: sl = best + gp
                if h[bp] >= sl:
                    pnls.append((sl - entry) * direction)
                    exited = True; break
        if not exited:
            eb = min(pos + max_bars, len(df) - 1)
            pnls.append((c[eb] - entry) * direction)

    pnls = np.array(pnls)
    if len(pnls) == 0:
        return {"total_trades": 0}
    return {
        "total_trades": len(pnls),
        "total_net_profit": float(pnls.sum()),
        "percent_profitable": float(np.mean(pnls > 0) * 100),
        "profit_factor": float(abs(pnls[pnls > 0].sum() / max(pnls[pnls < 0].sum(), 1e-12))),
    }


def main():
    ap = argparse.ArgumentParser(description="V2+z MT5 backtest runner")
    ap.add_argument("--pairs", nargs="+", default=PAIRS)
    ap.add_argument("--z", type=float, default=2.5)
    ap.add_argument("--stop", type=float, default=3.0)
    ap.add_argument("--trig", type=float, default=0.5)
    ap.add_argument("--gap", type=float, default=0.1)
    ap.add_argument("--start", default="2025.01.01")
    ap.add_argument("--end", default="2025.06.01")
    ap.add_argument("--run", action="store_true", default=True)
    ap.add_argument("--analyze", action="store_true", help="Re-parse existing reports")
    ap.add_argument("--compare", action="store_true", help="Compare MT5 vs Python")
    args = ap.parse_args()

    print(f"V2+z CPPF — MT5 vs Python Comparison")
    print(f"=====================================")
    print(f"Pairs: {args.pairs}")
    print(f"Params: z>={args.z} stop={args.stop} trig={args.trig} gap={args.gap}")
    print(f"Period: {args.start} → {args.end}")
    print()

    mt5_results = {}
    for pair in args.pairs:
        print(f"\n--- {pair} ---")

        # Create config and run
        ini_path, report_path = make_config(pair, z_thresh=args.z,
            stop_a=args.stop, trig_a=args.trig, gap_a=args.gap,
            start=args.start, end=args.end)

        if args.run:
            run_backtest(ini_path)

        # Parse results
        report = parse_report(report_path)
        if not report:
            print(f"  No report found at {report_path}")
            continue

        n_trades = report.get("_n_trades", 0)
        total_pnl = float(report.get("total_net_profit", 0))
        wr = float(report.get("percent_profitable", 0))
        pf = float(report.get("profit_factor", 0))
        dd = float(report.get("max_drawdown_percent", 0))

        print(f"  MT5: n={n_trades:>5d}  WR={wr:>5.1f}%  PnL={total_pnl:>+8.2f}  PF={pf:>6.2f}  DD={dd:>5.1f}%")
        mt5_results[pair] = report

        # Compare with Python
        if args.compare:
            py = run_python_bt(pair, z_thresh=args.z)
            py_n = py.get("total_trades", 0)
            py_pnl = py.get("total_net_profit", 0)
            py_wr = py.get("percent_profitable", 0)
            print(f"  PY:  n={py_n:>5d}  WR={py_wr:>5.1f}%  PnL={py_pnl:>+8.2f}")
            if n_trades > 0 and py_n > 0:
                diff_pnl = total_pnl - py_pnl
                pct = (total_pnl / abs(py_pnl) - 1) * 100 if abs(py_pnl) > 0.01 else 0
                print(f"  GAP: PnL diff={diff_pnl:>+8.2f} ({pct:>+.0f}%)")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'Pair':>10s} {'MT5_n':>7s} {'MT5_WR':>8s} {'MT5_PnL':>10s} {'PY_n':>7s} {'PY_WR':>8s} {'PY_PnL':>10s} {'GAP':>10s}")
    print(f"{'-'*10} {'-'*7} {'-'*8} {'-'*10} {'-'*7} {'-'*8} {'-'*10} {'-'*10}")
    for pair in args.pairs:
        mt5 = mt5_results.get(pair, {})
        mt5_n = mt5.get("_n_trades", 0)
        mt5_wr = mt5.get("percent_profitable", "?")
        mt5_pnl = mt5.get("total_net_profit", "?")
        if args.compare:
            py = run_python_bt(pair, z_thresh=args.z)
            py_n = py.get("total_trades", 0)
            py_wr = f"{py.get('percent_profitable', 0):.1f}"
            py_pnl = f"{py.get('total_net_profit', 0):.2f}"
            mt5_pnl_f = float(mt5_pnl) if isinstance(mt5_pnl, (int, float)) else 0
            py_pnl_f = float(py.get("total_net_profit", 0))
            gap = f"{mt5_pnl_f - py_pnl_f:>+8.2f}"
        else:
            py_n = "?"
            py_wr = "?"
            py_pnl = "?"
            gap = "?"
        print(f"{pair:>10s} {mt5_n:>7d} {mt5_wr:>8s} {mt5_pnl:>10s} {py_n!s:>7s} {py_wr:>8s} {py_pnl:>10s} {gap:>10s}")


if __name__ == "__main__":
    main()
