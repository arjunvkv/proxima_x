"""
P3.2: Boot-time Universe Integrity Audit.

Validates all data sources for the full canonical symbol universe
before the main engine loop starts. Prevents runtime surprises.
"""
import os
import sys
import numpy as np

SYMBOLS = ["EURJPY", "EURUSD", "GBPJPY", "USDJPY", "XAUUSD"]
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
M5_DIR = os.path.join(_PROJECT_ROOT, "data", "intraday")
TICK_DIR = os.path.join(_PROJECT_ROOT, "data", "ticks")
REQUIRED_M5_COLS = {"timestamp", "open", "high", "low", "close", "volume"}
MIN_M5_BARS = 1000
MIN_TICK_ROWS = 10000


def _load_parquet(path):
    try:
        import pyarrow.parquet as pq
        t = pq.read_table(path)
        return t
    except Exception:
        import pandas as pd
        try:
            df = pd.read_parquet(path)
            import pyarrow as pa
            return pa.Table.from_pandas(df)
        except Exception:
            return None


def _check_m5(symbol):
    path = f"{M5_DIR}/{symbol}_M5.parquet"
    if not os.path.isfile(path):
        return "FAIL", f"M5 parquet not found at {path}"
    t = _load_parquet(path)
    if t is None:
        return "FAIL", "M5 parquet unreadable"
    if t.num_rows < MIN_M5_BARS:
        return "FAIL", f"M5 has {t.num_rows} rows, need >= {MIN_M5_BARS}"
    cols = set(t.column_names)
    missing = REQUIRED_M5_COLS - cols
    if missing:
        return "FAIL", f"M5 missing columns: {missing}"
    ts_col = t.column("timestamp" if "timestamp" in t.column_names else "time")
    ts = ts_col.to_pylist()
    if len(ts) >= 2:
        sample = ts[:min(len(ts), 100)]
        diffs = [sample[i+1] - sample[i] for i in range(len(sample)-1)]
        if not (all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs)):
            return "FAIL", "M5 timestamps not monotonic"
    return "OK", None


def _check_ticks(symbol):
    path = f"{TICK_DIR}/{symbol}_ticks.parquet"
    if not os.path.isfile(path):
        return "WARN", f"Tick parquet not found at {path} (optional for synthetic)"
    t = _load_parquet(path)
    if t is None:
        return "WARN", "Tick parquet unreadable"
    if t.num_rows < MIN_TICK_ROWS:
        return "WARN", f"Ticks has {t.num_rows} rows, need >= {MIN_TICK_ROWS}"
    ts_name = "timestamp" if "timestamp" in t.column_names else "time"
    if ts_name in t.column_names:
        ts = t.column(ts_name).to_pylist()
        if len(ts) >= 2:
            sample = ts[:min(len(ts), 100)]
            diffs = [sample[i+1] - sample[i] for i in range(len(sample)-1)]
            if not (all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs)):
                return "WARN", "Tick timestamps not monotonic"
    return "OK", None


def _check_vpl(symbol):
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from deployment.get_vpl_signal import get_current_signal
        sig = get_current_signal(symbol)
        if sig is None:
            return "WARN", "VPL returned None (insufficient bars?)"
        regime = sig.get("regime", "")
        if "CORRUPT" in regime or "NO_DATA" in regime:
            return "WARN", f"VPL regime: {regime}"
    except Exception as e:
        return "WARN", f"VPL engine error: {e}"
    return "OK", None


def run_integrity_audit(skip_vpl: bool = False) -> dict:
    """Run full boot-time audit across all symbols/datasources.

    Args:
        skip_vpl: If True, skip the VPL engine check (heavy, ~20s per symbol).

    Returns:
        {
            "status": "CLEAN" | "DEGRADED" | "ABORT",
            "results": { symbol: { "m5": ..., "ticks": ..., "vpl": ... } },
            "messages": [ ... ],
        }
    """
    results = {}
    messages = []
    any_fail = False
    any_warn = False

    for sym in SYMBOLS:
        sym_result = {}
        m5_status, m5_msg = _check_m5(sym)
        sym_result["m5"] = m5_status
        if m5_status == "FAIL":
            any_fail = True
            messages.append(f"{sym} M5: {m5_msg}")
        elif m5_msg:
            messages.append(f"{sym} M5: {m5_msg}")

        tick_status, tick_msg = _check_ticks(sym)
        sym_result["ticks"] = tick_status
        if tick_status == "WARN":
            any_warn = True
            messages.append(f"{sym} ticks: {tick_msg}")
        elif tick_msg:
            messages.append(f"{sym} ticks: {tick_msg}")

        if skip_vpl:
            sym_result["vpl"] = "SKIP"
        else:
            vpl_status, vpl_msg = _check_vpl(sym)
            sym_result["vpl"] = vpl_status
            if vpl_status == "WARN":
                any_warn = True
                messages.append(f"{sym} VPL: {vpl_msg}")
            elif vpl_msg:
                messages.append(f"{sym} VPL: {vpl_msg}")

        results[sym] = sym_result

    if any_fail:
        status = "ABORT"
    elif any_warn:
        status = "DEGRADED"
    else:
        status = "CLEAN"

    return {"status": status, "results": results, "messages": messages}


def audit_summary(audit_result: dict) -> str:
    """Pretty-print the audit results."""
    lines = []
    lines.append("  BOOT INTEGRITY AUDIT")
    lines.append("-" * 52)
    for sym in SYMBOLS:
        r = audit_result["results"].get(sym, {})
        parts = []
        for source in ["m5", "ticks", "vpl"]:
            s = r.get(source, "?")
            if s == "OK":
                parts.append(f"{source}=OK")
            elif s == "WARN":
                parts.append(f"{source}=WARN")
            elif s == "SKIP":
                parts.append(f"{source}=SKIP")
            else:
                parts.append(f"{source}=FAIL")
        ok_vals = {"OK", "SKIP"}
        status = "OK" if all(v in ok_vals for v in r.values()) else (
            "WARN" if any(v == "WARN" for v in r.values()) else "FAIL"
        )
        label = {"OK": "OK", "WARN": "WARN", "FAIL": "FAIL"}.get(status, "?")
        lines.append(f"  {sym:<10s} {label:<5s} ({', '.join(parts)})")

    lines.append(f"  Universe Status: {audit_result['status']}")
    if audit_result["status"] == "ABORT":
        lines.append("  Trade Mode:      ABORT — M5 data failure")
    elif audit_result["status"] == "DEGRADED":
        lines.append("  Trade Mode:      ALLOWED (with warnings)")
    else:
        lines.append("  Trade Mode:      FULL")

    if audit_result["messages"]:
        lines.append("")
        for m in audit_result["messages"]:
            lines.append(f"    {m}")

    return "\n".join(lines)
